use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

fn make_indices(pc_char: u8, rank: u8, file: u8) -> (usize, usize) {
    const PIECES: &[u8] = b"PNBRQKpnbrqk";

    let pc = PIECES
        .iter()
        .position(|&char| char == pc_char)
        .unwrap_or_else(|| panic!("Unrecognized piece"));
    let sq = ((rank << 3) ^ file) as usize;
    (
        64 * pc + sq,
        64 * ((pc + 6) % 12) + sq ^ 56
    )
}

#[derive(PartialEq)]
enum Color {
    White,
    Black,
}

#[pyfunction]
fn fen_to_indices(fen: &str) -> PyResult<(Vec<usize>, Vec<usize>)> {
    let mut stm_indices = Vec::with_capacity(32);
    let mut nstm_indices = Vec::with_capacity(32);
    let mut rank = 7;
    let mut file = 0;

    let mut parts = fen.split_whitespace();

    let position = match parts.next() {
        Some(s) => String::from(s),
        None => panic!("Malformed FEN."),
    };

    let stm = match parts.next() {
        Some("w") => Color::White,
        Some("b") => Color::Black,
        _ => panic!("Unrecognized color.")
    };

    for &char in position.as_bytes() {
        match char {
            b'A'..=b'z' => {
                let (w_index, b_index) = make_indices(char, rank, file);
                assert!(w_index < 768 && b_index < 768);
                match stm {
                    Color::White => {
                        stm_indices.push(w_index);
                        nstm_indices.push(b_index);
                    }
                    Color::Black => {
                        stm_indices.push(b_index);
                        nstm_indices.push(w_index);
                    }
                }
                file += 1;
            }
            b'/' => {
                rank -= 1;
                file = 0;
            }
            b'1'..=b'8' => {
                file += char - b'0';
            }
            _ => {
                panic!("Unrecognized character.");
            }
        }
    }

    stm_indices.resize(32, 768);
    nstm_indices.resize(32, 768);

    Ok((stm_indices, nstm_indices))
}

#[pyfunction]
fn indices_to_vec(indices: Vec<usize>) -> PyResult<[f32; 768]> {
    let mut input = [0.0; 768];

    for idx in indices {
        input[idx] = 1.0
    }

    Ok(input)
}

#[pymodule]
fn fen_parser(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fen_to_indices, m)?)?;
    m.add_function(wrap_pyfunction!(indices_to_vec, m)?)?;
    Ok(())
}
