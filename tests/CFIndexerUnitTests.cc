#include "hpg_indexing.hpp"
#include "gtest/gtest.h"

TEST(CFSimpleIndexer, Extents) {

  const unsigned nb = 1;
  const unsigned nt = 2;
  const unsigned nw = 3;
  const unsigned nf = 4;
  const unsigned nm = 5;

  for (bool bv : {false, true}) {
    unsigned plane = 1, grp = 1;
    if (bv) grp = nb; else plane = nb;
    for (bool tv : {false, true}) {
      if (tv) grp *= nt; else plane *= nt;
      for (bool wv : {false, true}) {
        if (wv) grp *= nw; else plane *= nw;
        for (bool fv : {false, true}) {
          if (fv) grp *= nf; else plane *= nf;
          for (bool mv : {false, true}) {
            if (mv) grp *= nm; else plane *= nm;
            hpg::CFSimpleIndexer
              indexer({nb, bv}, {nt, tv}, {nw, wv}, {nf, fv}, {nm, mv});
            EXPECT_EQ(
              indexer.cf_extents(),
              hpg::CFSimpleIndexer::cf_index_t({plane, grp}));
            if (mv) grp /= nm; else plane /= nm;
          }
          if (fv) grp /= nf; else plane /= nf;
        }
        if (wv) grp /= nw; else plane /= nw;
      }
      if (tv) grp /= nt; else plane /= nt;
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
    indexer({nb, true}, {nt, false}, {nw, true}, {nf, true}, {nm, false});
  EXPECT_EQ(
    indexer.cf_index(hpg::CFCellIndex(0, 0, 0, 0, 0)),
    hpg::CFSimpleIndexer::cf_index_t({0, 0}));
  EXPECT_EQ(
    hpg::CFCellIndex(0, 0, 0, 0, 0),
    indexer.cell_index({0, 0}));
  EXPECT_EQ(
    indexer.cf_index(hpg::CFCellIndex(1, 0, 0, 0, 0)),
    hpg::CFSimpleIndexer::cf_index_t({0, nw * nf}));
  EXPECT_EQ(
    hpg::CFCellIndex(1, 0, 0, 0, 0),
    indexer.cell_index({0, nw * nf}));
  EXPECT_EQ(
    indexer.cf_index(hpg::CFCellIndex(1, 0, 2, 0, 0)),
    hpg::CFSimpleIndexer::cf_index_t({0, nw * nf + 2 * nf}));
  EXPECT_EQ(
    hpg::CFCellIndex(1, 0, 2, 0, 0),
    indexer.cell_index({0, nw * nf + 2 * nf}));
  EXPECT_EQ(
    indexer.cf_index(hpg::CFCellIndex(1, 0, 2, 0, 1)),
    hpg::CFSimpleIndexer::cf_index_t({1, nw * nf + 2 * nf}));
  EXPECT_EQ(
    hpg::CFCellIndex(1, 0, 2, 0, 1),
    indexer.cell_index({1, nw * nf + 2 * nf}));
  EXPECT_EQ(
    indexer.cf_index(hpg::CFCellIndex(1, 1, 2, 0, 0)),
    hpg::CFSimpleIndexer::cf_index_t({nm, nw * nf + 2 * nf}));
  EXPECT_EQ(
    hpg::CFCellIndex(1, 1, 2, 0, 0),
    indexer.cell_index({nm, nw * nf + 2 * nf}));
  EXPECT_EQ(
    indexer.cf_index(hpg::CFCellIndex(1, 1, 2, 0, 1)),
    hpg::CFSimpleIndexer::cf_index_t({nm + 1, nw * nf + 2 * nf}));
  EXPECT_EQ(
    hpg::CFCellIndex(1, 1, 2, 0, 1),
    indexer.cell_index({nm + 1, nw * nf + 2 * nf}));
  EXPECT_EQ(
    indexer.cf_index(hpg::CFCellIndex(1, 1, 2, 3, 1)),
    hpg::CFSimpleIndexer::cf_index_t({nm + 1, nw * nf + 2 * nf + 3}));
  EXPECT_EQ(
    hpg::CFCellIndex(1, 1, 2, 3, 1),
    indexer.cell_index({nm + 1, nw * nf + 2 * nf + 3}));
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
