use crate::fields::Field64;
use crate::hash::Hasher;

use super::Commitment;

///
#[derive(Debug)]
pub struct MerkleCap<F: Field64, H: Hasher<F>>(pub Vec<H::Hash>);

///
#[derive(Debug)]
pub struct MerkleTree<F: Field64, H: Hasher<F>> {
    /// The data in the leaves of the Merkle tree.
    pub leaves: Vec<Vec<F>>,

    /// The digests in the tree. Consists of `cap.len()` sub-trees, each corresponding to one
    /// element in `cap`. Each subtree is contiguous and located at
    /// `digests[digests.len() / cap.len() * i..digests.len() / cap.len() * (i + 1)]`.
    /// Within each subtree, siblings are stored next to each other. The layout is,
    /// left_child_subtree || left_child_digest || right_child_digest || right_child_subtree, where
    /// left_child_digest and right_child_digest are H::Hash and left_child_subtree and
    /// right_child_subtree recurse. Observe that the digest of a node is stored by its _parent_.
    /// Consequently, the digests of the roots are not stored here (they can be found in `cap`).
    pub digests: Vec<H::Hash>,

    /// The Merkle cap.
    pub cap: MerkleCap<F, H>,
}

///
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct MerkleCommitment;

impl<F: Field64, H: Hasher<F>> Commitment<F, H> for MerkleCommitment {}
