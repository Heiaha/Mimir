use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

fn make_index(pc_char: u8, rank: u8, file: u8) -> usize {
    const PIECES: &[u8] = "PNBRQKpnbrqk".as_bytes();

    let pc = PIECES
        .iter()
        .position(|&char| char == pc_char)
        .unwrap_or_else(|| panic!("Unrecognized piece"));
    let sq = ((rank << 3) ^ file) as usize;
    64 * pc + sq
}

#[pyfunction]
fn fen_to_vec(fen: &str) -> PyResult<Vec<usize>> {
    let mut indices = Vec::with_capacity(32);
    let mut rank = 7;
    let mut file = 0;

    let mut parts = fen.split_whitespace();

    let position = match parts.next() {
        Some(s) => String::from(s),
        None => panic!("Malformed FEN."),
    };
    for &char in position.as_bytes() {
        match char {
            b'A'..=b'z' => {
                indices.push(make_index(char, rank, file));
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

    Ok(indices)
}

#[pymodule]
fn fen_parser(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fen_to_vec, m)?)?;
    Ok(())
}
