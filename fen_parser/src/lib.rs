use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

const PIECES: &[u8] = b"PNBRQKpnbrqk";
const WHITE_KING: usize = 5;
const BLACK_KING: usize = 11;

const N_BASE_FEATURES: usize = 768;
const N_KING_BUCKETS: usize = 4;
const N_FEATURES: usize = N_KING_BUCKETS * N_BASE_FEATURES;

fn king_bucket(ksq_norm: usize) -> usize {
    match ksq_norm >> 3 {
        0 => 0,
        1 => 1,
        2 | 3 => 2,
        _ => 3,
    }
}

#[derive(PartialEq)]
enum Color {
    White,
    Black,
}

// A perspective's feature transform, the counterpart of FeatureCtx in
// Weiawaga's nnue.rs: piece colors and squares are color-relative (rank
// flip `^ 56` for Black), the board is horizontally mirrored (`^ 7`) while
// the perspective's OWN king is on files e-h, and the normalized king
// square selects a bucket's block of features.
struct Perspective {
    sq_xor: usize,
    pc_flip: usize,
    offset: usize,
}

impl Perspective {
    fn new(color: Color, own_king_sq: usize) -> Self {
        let (rel_xor, pc_flip) = match color {
            Color::White => (0, 0),
            Color::Black => (56, 6),
        };
        let ksq_rel = own_king_sq ^ rel_xor;
        let mirror_xor = if ksq_rel & 7 >= 4 { 7 } else { 0 };

        Self {
            sq_xor: rel_xor ^ mirror_xor,
            pc_flip,
            offset: N_BASE_FEATURES * king_bucket(ksq_rel ^ mirror_xor),
        }
    }

    fn feature_index(&self, pc: usize, sq: usize) -> usize {
        let index = self.offset + 64 * ((pc + self.pc_flip) % 12) + (sq ^ self.sq_xor);
        assert!(index < N_FEATURES);
        index
    }
}

fn parse_indices(fen: &str) -> (Vec<usize>, Vec<usize>) {
    let mut pieces: Vec<(usize, usize)> = Vec::with_capacity(32);
    let mut rank: u8 = 7;
    let mut file: u8 = 0;

    let mut parts = fen.split_whitespace();

    let position = parts.next().expect("Malformed FEN.");

    let stm = match parts.next() {
        Some("w") => Color::White,
        Some("b") => Color::Black,
        _ => panic!("Unrecognized color."),
    };

    for &char in position.as_bytes() {
        match char {
            b'A'..=b'z' => {
                let pc = PIECES
                    .iter()
                    .position(|&pc_char| pc_char == char)
                    .unwrap_or_else(|| panic!("Unrecognized piece"));
                let sq = ((rank << 3) ^ file) as usize;
                pieces.push((pc, sq));
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

    let king_sq = |king: usize| {
        pieces
            .iter()
            .find_map(|&(pc, sq)| (pc == king).then_some(sq))
            .expect("FEN has no king.")
    };

    let white = Perspective::new(Color::White, king_sq(WHITE_KING));
    let black = Perspective::new(Color::Black, king_sq(BLACK_KING));
    let (own, opp) = match stm {
        Color::White => (white, black),
        Color::Black => (black, white),
    };

    let mut stm_indices = Vec::with_capacity(32);
    let mut nstm_indices = Vec::with_capacity(32);

    for &(pc, sq) in &pieces {
        stm_indices.push(own.feature_index(pc, sq));
        nstm_indices.push(opp.feature_index(pc, sq));
    }

    stm_indices.resize(32, N_FEATURES);
    nstm_indices.resize(32, N_FEATURES);

    (stm_indices, nstm_indices)
}

#[pyfunction]
fn fen_to_indices(fen: &str) -> PyResult<(Vec<usize>, Vec<usize>)> {
    Ok(parse_indices(fen))
}

#[pyfunction]
fn indices_to_vec(indices: Vec<usize>) -> PyResult<[f32; N_FEATURES]> {
    let mut input = [0.0; N_FEATURES];

    for idx in indices {
        input[idx] = 1.0
    }

    Ok(input)
}

#[pymodule]
fn fen_parser(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fen_to_indices, m)?)?;
    m.add_function(wrap_pyfunction!(indices_to_vec, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Contract test with Weiawaga's nnue.rs (feature_indices_match_trainer
    // _convention): same position, same expected indices, hardcoded in both
    // repos. If either side changes its feature convention, both tests must
    // be updated together.
    #[test]
    fn feature_indices_match_engine_convention() {
        // White king b1 (files a-d, not mirrored), black king g8 (mirrored
        // to b1); both perspectives in bucket 0. FEN scan order: black king
        // first.
        let (stm, nstm) = parse_indices("6k1/8/8/8/8/8/8/1K6 w - - 0 1");

        assert_eq!(&stm[..2], &[766, 321]); // white perspective
        assert_eq!(&nstm[..2], &[321, 766]); // black perspective
        assert!(
            stm[2..]
                .iter()
                .chain(nstm[2..].iter())
                .all(|&i| i == N_FEATURES)
        );
    }

    #[test]
    fn bucket_offsets_match_engine_convention() {
        // White king d5 (rank 5, bucket 3), black king g8 (mirrored to b1,
        // bucket 0).
        let (stm, nstm) = parse_indices("6k1/8/8/3K4/8/8/8/8 w - - 0 1");

        assert_eq!(&stm[..2], &[3070, 2659]); // white perspective
        assert_eq!(&nstm[..2], &[321, 732]); // black perspective
    }

    #[test]
    fn startpos_perspectives_are_symmetric() {
        // The starting position is symmetric between the two perspectives
        // (both kings on the e-file, so both perspectives mirrored), which
        // the feature transform must preserve.
        let (stm, nstm) =
            parse_indices("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        let mut stm_sorted = stm.clone();
        let mut nstm_sorted = nstm.clone();
        stm_sorted.sort_unstable();
        nstm_sorted.sort_unstable();
        assert_eq!(stm_sorted, nstm_sorted);
    }
}
