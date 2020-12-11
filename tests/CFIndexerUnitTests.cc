#include "hpg_indexing.hpp"
#include "gtest/gtest.h"

TEST(CFSimpleIndexer, Extents) {

  const unsigned nb = 2;
  const unsigned nt = 3;
  const unsigned nw = 4;
  const unsigned nf = 5;
  const unsigned np = 6;

  for (bool bv : {false, true}) {
    unsigned cube = 1, supp = 1;
    if (bv) supp = nb; else cube = nb;
    for (bool tv : {false, true}) {
      if (tv) supp *= nt; else cube *= nt;
      for (bool wv : {false, true}) {
        if (wv) supp *= nw; else cube *= nw;
        for (bool fv : {false, true}) {
          if (fv) supp *= nf; else cube *= nf;
          for (bool pv : {false, true}) {
            if (pv) supp *= np; else cube *= np;
            hpg::CFSimpleIndexer
              indexer({nb, bv}, {nt, tv}, {nw, wv}, {nf, fv}, {np, pv});
            EXPECT_EQ(indexer.extents(), hpg::vis_cf_index_t(cube, supp));
            if (pv) supp /= np; else cube /= np;
          }
          if (fv) supp /= nf; else cube /= nf;
        }
        if (wv) supp /= nw; else cube /= nw;
      }
      if (tv) supp /= nt; else cube /= nt;
    }
  }
}

TEST(CFSimpleIndexer, Examples) {

  const unsigned nb = 2;
  const unsigned nt = 2;
  const unsigned nw = 3;
  const unsigned nf = 4;
  const unsigned np = 2;

  hpg::CFSimpleIndexer
    indexer({nb, true}, {nt, false}, {nw, true}, {nf, true}, {np, false});
  EXPECT_EQ(
    indexer.cf_index(hpg::CFCellIndex(0, 0, 0, 0, 0)),
    hpg::vis_cf_index_t(0, 0));
  EXPECT_EQ(
    hpg::CFCellIndex(0, 0, 0, 0, 0),
    indexer.cell_index({0, 0}));
  EXPECT_EQ(
    indexer.cf_index(hpg::CFCellIndex(1, 0, 0, 0, 0)),
    hpg::vis_cf_index_t(0, nw * nf));
  EXPECT_EQ(
    hpg::CFCellIndex(1, 0, 0, 0, 0),
    indexer.cell_index({0, nw * nf}));
  EXPECT_EQ(
    indexer.cf_index(hpg::CFCellIndex(1, 0, 2, 0, 0)),
    hpg::vis_cf_index_t(0, nw * nf + 2 * nf));
  EXPECT_EQ(
    hpg::CFCellIndex(1, 0, 2, 0, 0),
    indexer.cell_index({0, nw * nf + 2 * nf}));
  EXPECT_EQ(
    indexer.cf_index(hpg::CFCellIndex(1, 0, 2, 0, 1)),
    hpg::vis_cf_index_t(1, nw * nf + 2 * nf));
  EXPECT_EQ(
    hpg::CFCellIndex(1, 0, 2, 0, 1),
    indexer.cell_index({1, nw * nf + 2 * nf}));
  EXPECT_EQ(
    indexer.cf_index(hpg::CFCellIndex(1, 1, 2, 0, 0)),
    hpg::vis_cf_index_t(np, nw * nf + 2 * nf));
  EXPECT_EQ(
    hpg::CFCellIndex(1, 1, 2, 0, 0),
    indexer.cell_index({np, nw * nf + 2 * nf}));
  EXPECT_EQ(
    indexer.cf_index(hpg::CFCellIndex(1, 1, 2, 0, 1)),
    hpg::vis_cf_index_t(np + 1, nw * nf + 2 * nf));
  EXPECT_EQ(
    hpg::CFCellIndex(1, 1, 2, 0, 1),
    indexer.cell_index({np + 1, nw * nf + 2 * nf}));
  EXPECT_EQ(
    indexer.cf_index(hpg::CFCellIndex(1, 1, 2, 3, 1)),
    hpg::vis_cf_index_t(np + 1, nw * nf + 2 * nf + 3));
  EXPECT_EQ(
    hpg::CFCellIndex(1, 1, 2, 3, 1),
    indexer.cell_index({np + 1, nw * nf + 2 * nf + 3}));
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
