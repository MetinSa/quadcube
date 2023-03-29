use numpy::{
    ndarray::{Array2, Dim},
    IntoPyArray, PyArray, PyReadonlyArray1,
};
use pyo3::{
    prelude::{pymodule, PyModule, PyResult, Python},
    FromPyObject,
};

#[pymodule]
fn quadcube(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[derive(FromPyObject)]
    enum Pixels<'py> {
        Int(i64),
        List(Vec<i64>),
        NDArray(PyReadonlyArray1<'py, i64>),
    }

    /// Converts Quadcube resolution 15 pixel numbers to ecliptic unit vectors.
    /// The input can be a single pixel number, a list of pixel numbers, or a
    /// numpy array of pixel numbers.
    /// The output is a 3xN numpy array of ecliptic unit vectors.
    /// 
    /// Parameters
    /// ----------
    /// ipix : int, list, or numpy array
    ///    Pixel number(s) to convert to unit vectors.
    ///     
    /// Returns
    /// -------
    /// unit_vectors : numpy array
    ///   3xN numpy array of unit vectors.
    #[pyfn(m)]
    fn pix2vec<'py>(py: Python<'py>, ipix: Pixels<'py>) -> &'py PyArray<f64, Dim<[usize; 2]>> {
        let pixels = match ipix {
            Pixels::Int(p) => vec![p],
            Pixels::List(p) => p,
            Pixels::NDArray(p) => p.as_array().to_vec(),
        };

        let mut unit_vectors = Array2::<f64>::zeros((3, pixels.len()));
        for (idx, pixel) in pixels.iter().enumerate() {
            let unit_vector = rust_fn::pixel_to_unit_vector(&(*pixel as usize));
            unit_vectors[[0, idx]] = unit_vector[0];
            unit_vectors[[1, idx]] = unit_vector[1];
            unit_vectors[[2, idx]] = unit_vector[2];
        }
        unit_vectors.into_pyarray(py)
    }

    Ok(())
}

mod rust_fn {
    use cached::proc_macro::cached;

    const P: [f64; 28] = [
        -0.27292696,
        -0.07629969,
        -0.02819452,
        -0.22797056,
        -0.01471565,
        0.27058160,
        0.54852384,
        0.48051509,
        -0.56800938,
        -0.60441560,
        -0.62930065,
        -1.74114454,
        0.30803317,
        1.50880086,
        0.93412077,
        0.25795794,
        1.71547508,
        0.98938102,
        -0.93678576,
        -1.41601920,
        -0.63915306,
        0.02584375,
        -0.53022337,
        -0.83180469,
        0.08693841,
        0.33887446,
        0.52032238,
        0.14381585,
    ];

    const G0: f64 = 1.37484847732;
    const G0_1: f64 = 1.0 - 1.37484847732;
    const G: f64 = -0.13161671474;
    const MG: f64 = 0.004869491981 - G;
    const W1: f64 = -0.159596235474;
    const C00: f64 = 0.141189631152;
    const C10: f64 = 0.0809701286525;
    const C01: f64 = -0.281528535557;
    const C11: f64 = 0.15384112876;
    const C20: f64 = -0.178251207466;
    const C02: f64 = 0.106959469314;
    const D0: f64 = 0.0759196200467;
    const D1: f64 = -0.0217762490699;

    const IT28: usize = 2_usize.pow(28);
    const N_10: usize = 1024;

    pub fn pixel_to_unit_vector(pixel: &usize) -> [f64; 3] {
        let (i_x, i_y) = get_ix_iy();

        let n_face = pixel / IT28;
        let mut n = pixel % IT28;
        let i = n % N_10;
        n /= N_10;
        let j = n % N_10;
        let k = n / N_10;
        let j_x = N_10 * i_x[k] + 32 * i_x[j] + i_x[i];
        let j_y = N_10 * i_y[k] + 32 * i_y[j] + i_y[i];

        let x = (j_x as f64 - 8191.5) / 8192.0;
        let y = (j_y as f64 - 8191.5) / 8192.0;

        let mut x_i = deproject_face(x, y);
        let mut eta = deproject_face(y, x);

        for _ in 0..2 {
            let x_p = project_face(x_i, eta);
            let y_p = project_face(eta, x_i);
            x_i -= x_p - x;
            eta -= y_p - y;
        }
        cube_face_to_vec(n_face, x_i, eta)
    }

    fn cube_face_to_vec(n_face: usize, x_i: f64, eta: f64) -> [f64; 3] {
        let x_i_1 = x_i.abs().max(eta.abs());
        let eta_1 = x_i.abs().min(eta.abs());
        let norm = 1.0 / (1.0 + x_i_1 * x_i_1 + eta_1 * eta_1).sqrt();

        match n_face {
            0 => [-eta * norm, x_i * norm, norm],
            1 => [norm, x_i * norm, eta * norm],
            2 => [-x_i * norm, norm, eta * norm],
            3 => [-norm, -x_i * norm, eta * norm],
            4 => [x_i * norm, -norm, eta * norm],
            5 => [eta * norm, -x_i * norm, norm],
            _ => panic!("Invalid face number: {}", n_face),
        }
    }

    fn deproject_face(a: f64, b: f64) -> f64 {
        let a_squared = a * a;
        let b_squared = b * b;

        a * (1.0
            + (1.0 - a_squared)
                * (P[0]
                    + a_squared
                        * (P[1]
                            + a_squared
                                * (P[3]
                                    + a_squared
                                        * (P[6]
                                            + a_squared
                                                * (P[10]
                                                    + a_squared * (P[15] + a_squared * P[21])))))
                    + b_squared
                        * (P[2]
                            + a_squared
                                * (P[4]
                                    + a_squared
                                        * (P[7]
                                            + a_squared
                                                * (P[11]
                                                    + a_squared * (P[16] + a_squared * P[22]))))
                            + b_squared
                                * (P[5]
                                    + a_squared
                                        * (P[8]
                                            + a_squared
                                                * (P[12]
                                                    + a_squared * (P[17] + a_squared * P[23])))
                                    + b_squared
                                        * (P[9]
                                            + a_squared
                                                * (P[13]
                                                    + a_squared * (P[18] + a_squared * P[24]))
                                            + b_squared
                                                * (P[14]
                                                    + a_squared * (P[19] + a_squared * P[25])
                                                    + b_squared
                                                        * (P[20]
                                                            + a_squared * P[26]
                                                            + b_squared * P[27])))))))
    }

    fn project_face(a: f64, b: f64) -> f64 {
        let a_2 = a * a;
        let b_2 = b * b;
        let a_4 = a_2 * a_2;
        let b_4 = b_2 * b_2;
        let one_m_a_2 = 1.0 - a_2;
        let one_m_b_2 = 1.0 - b_2;
        let c0011 = C00 + C11 * a_2 * b_2;

        a * (G0
            + a_2 * G0_1
            + one_m_a_2
                * (b_2
                    * (G + MG * a_2
                        + one_m_b_2 * (c0011 + C10 * a_2 + C01 * b_2 + C20 * a_4 + C02 * b_4))
                    + a_2 * (W1 - one_m_a_2 * (D0 + D1 * a_2))))
    }

    #[cached]
    fn get_ix_iy() -> ([usize; N_10], [usize; N_10]) {
        let mut j_x = [0; N_10];
        let mut j_y = [0; N_10];

        let mut id: usize;
        for kpix in 0..N_10 {
            let mut jpix = kpix as usize;
            let mut ix = 0;
            let mut iy = 0;
            let mut ip = 1;
            while jpix != 0 {
                id = jpix % 2;
                jpix /= 2;
                ix += id * ip;
                id = jpix % 2;
                jpix /= 2;
                iy += id * ip;
                ip *= 2;
            }

            j_x[kpix] = ix;
            j_y[kpix] = iy;
        }
        (j_x, j_y)
    }
}
