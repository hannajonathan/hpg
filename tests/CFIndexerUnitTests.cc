#include "hpg_indexing.hpp"
#include "gtest/gtest.h"

TEST(CFSimpleIndexer, Extents) {

  const unsigned nb = 1;
  const unsigned nt = 2;
  const unsigned nw = 3;
  const unsigned nf = 4;
  const unsigned nm = 5;

  for (bool bv : {false, true}) {
    unsigned cube = 1, grp = 1;
    if (bv) grp = nb; else cube = nb;
    for (bool tv : {false, true}) {
      if (tv) grp *= nt; else cube *= nt;
      for (bool wv : {false, true}) {
        if (wv) grp *= nw; else cube *= nw;
        for (bool fv : {false, true}) {
          if (fv) grp *= nf; else cube *= nf;
          hpg::CFSimpleIndexer
            indexer({nb, bv}, {nt, tv}, {nw, wv}, {nf, fv}, nm);
          EXPECT_EQ(
            indexer.cf_extents(),
            hpg::CFSimpleIndexer::cf_index_t({nm, cube, grp}));
          if (fv) grp /= nf; else cube /= nf;
        }
        if (wv) grp /= nw; else cube /= nw;
      }
      if (tv) grp /= nt; else cube /= nt;
    }
  }
}

TEST(CFSimpleIndexer, Examples) {

  const unsigned nb = 2;
  const unsigned nt = 2;
  const unsigned nw = 3;
  const unsigned nf = 4;
  const unsigned nm = 8;

  hpg::CFSimpleIndexer
    indexer({nb, true}, {nt, false}, {nw, true}, {nf, true}, nm);
  EXPECT_EQ(
    indexer.cf_index(hpg::CFCellIndex(0, 0, 0, 0, 0)),
    hpg::CFSimpleIndexer::cf_index_t({0, 0, 0}));
  EXPECT_EQ(
    hpg::CFCellIndex(0, 0, 0, 0, 0),
    indexer.cell_index({0, 0, 0}));
  EXPECT_EQ(
    indexer.cf_index(hpg::CFCellIndex(1, 0, 0, 0, 0)),
    hpg::CFSimpleIndexer::cf_index_t({0, 0, nw * nf}));
  EXPECT_EQ(
    hpg::CFCellIndex(1, 0, 0, 0, 0),
    indexer.cell_index({0, 0, nw * nf}));
  EXPECT_EQ(
    indexer.cf_index(hpg::CFCellIndex(1, 0, 2, 0, 0)),
    hpg::CFSimpleIndexer::cf_index_t({0, 0, nw * nf + 2 * nf}));
  EXPECT_EQ(
    hpg::CFCellIndex(1, 0, 2, 0, 0),
    indexer.cell_index({0, 0, nw * nf + 2 * nf}));
  EXPECT_EQ(
    indexer.cf_index(hpg::CFCellIndex(1, 0, 2, 0, 1)),
    hpg::CFSimpleIndexer::cf_index_t({1, 0, nw * nf + 2 * nf}));
  EXPECT_EQ(
    hpg::CFCellIndex(1, 0, 2, 0, 1),
    indexer.cell_index({1, 0, nw * nf + 2 * nf}));
  EXPECT_EQ(
    indexer.cf_index(hpg::CFCellIndex(1, 1, 2, 0, 0)),
    hpg::CFSimpleIndexer::cf_index_t({0, 1, nw * nf + 2 * nf}));
  EXPECT_EQ(
    hpg::CFCellIndex(1, 1, 2, 0, 0),
    indexer.cell_index({0, 1, nw * nf + 2 * nf}));
  EXPECT_EQ(
    indexer.cf_index(hpg::CFCellIndex(1, 1, 2, 0, 1)),
    hpg::CFSimpleIndexer::cf_index_t({1, 1, nw * nf + 2 * nf}));
  EXPECT_EQ(
    hpg::CFCellIndex(1, 1, 2, 0, 1),
    indexer.cell_index({1, 1, nw * nf + 2 * nf}));
  EXPECT_EQ(
    indexer.cf_index(hpg::CFCellIndex(1, 1, 2, 3, 1)),
    hpg::CFSimpleIndexer::cf_index_t({1, 1, nw * nf + 2 * nf + 3}));
  EXPECT_EQ(
    hpg::CFCellIndex(1, 1, 2, 3, 1),
    indexer.cell_index({1, 1, nw * nf + 2 * nf + 3}));
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
