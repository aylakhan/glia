#include "hmt/bc_feat.hxx"
#include "hmt/tree_build.hxx"
#include "hmt/hmt_util.hxx"
#include "type/region_map.hxx"
#include "util/image_io.hxx"
#include "util/text_io.hxx"
#include "util/text_cmd.hxx"
#include "util/mp.hxx"
using namespace glia;
using namespace glia::hmt;

struct NodeData {
  Label label;
};

std::string segImageFile;
std::string mergeOrderFile;
std::string saliencyFile;
std::string pbImageFile;
std::vector<ImageFileHistPair> rbImageFiles;
std::vector<ImageFileHistPair> rlImageFiles;
std::vector<ImageFileHistPair> rImageFiles;
std::vector<ImageFileHistPair> bImageFiles;
std::string maskImageFile;
double initSal = 1.0;
double salBias = 1.0;
std::vector<double> boundaryThresholds;
bool normalizeShape = false;
bool useLogShape = false;
std::string featFile;


bool operation ()
{
  typedef TRegionMap<Label, Point<DIMENSION>> RegionMap;
  // Load and set up images
  auto segImage = readImage<LabelImage<DIMENSION>>(segImageFile);
  RealImage<DIMENSION>::Pointer pbImage;
  std::vector<ImageHistPair<RealImage<DIMENSION>::Pointer>>
      rImages, rlImages, bImages;
  prepareImages(
      pbImage, rImages, rlImages, bImages, pbImageFile, rbImageFiles,
      rlImageFiles, rImageFiles, bImageFiles);
  auto mask = maskImageFile.empty()?
      LabelImage<DIMENSION>::Pointer(nullptr):
      readImage<LabelImage<DIMENSION>>(maskImageFile);
  // Set up normalizing area/length
  double normalizingArea =
      normalizeShape ? getImageVolume(segImage) : 1.0;
  double normalizingLength =
      normalizeShape ? getImageDiagonal(segImage) : 1.0;
  // Set up regions etc.
  std::vector<TTriple<Label>> order;
  readData(order, mergeOrderFile, true);
  std::vector<double> saliencies;
  std::unordered_map<Label, double> saliencyMap;
  if (!saliencyFile.empty()) {
    readData(saliencies, saliencyFile, true);
    genSaliencyMap(saliencyMap, order, saliencies, initSal, salBias);
  }
  RegionMap rmap(segImage, mask, order, false);
  // Generate region features
  int rn = rmap.size();
  std::vector<RegionFeats> rfeats(rn);
  parfor(rmap, true, [&rfeats, &rImages, &rlImages, &bImages, &pbImage,
                      &saliencyMap, normalizingArea, normalizingLength]
         (RegionMap::const_iterator rit, int i) {
           rfeats[i].generate(
               rit->second, normalizingArea, normalizingLength,
               pbImage, boundaryThresholds, rImages, rlImages, bImages,
               ccpointer(saliencyMap, rit->first));
         }, 0);
  std::unordered_map<Label, RegionFeats const*> rfmap;
  int ri = 0;
  for (auto const& rp: rmap) { rfmap[rp.first] = &rfeats[ri++]; }
  // Generate boundary features
  int bn = order.size();
  std::vector<BoundaryFeats> bfeats(bn);
  parfor(0, bn, true, [&order, &rmap, &rfeats, &rfmap, &bfeats,
                       &pbImage, &bImages, normalizingLength](int i) {
      Label r0 = order[i].x0;
      Label r1 = order[i].x1;
      Label r2 = order[i].x2;
      auto const* rp0 = rfmap.find(r0)->second;
      auto const* rp1 = rfmap.find(r1)->second;
      auto const* rp2 = rfmap.find(r2)->second;
      // Keep region 0 area <= region 1 area
      if (rp0->shape->area > rp1->shape->area) {
        std::swap(r0, r1);
        std::swap(rp0, rp1);
      }
      RegionMap::Region::Boundary b;
      getBoundary(b, rmap.find(r0)->second, rmap.find(r1)->second);
      bfeats[i].generate(
          b, normalizingLength, *rp0, *rp1, *rp2, pbImage,
          boundaryThresholds, bImages);
    }, 0);
  // Log shape
  if (useLogShape) {
    parfor(0, rn, false, [&rfeats](int i) { rfeats[i].log(); }, 0);
    parfor(0, bn, false, [&bfeats](int i) { bfeats[i].log(); }, 0);
  }
  std::unordered_map<Label, BoundaryFeats const*> bfmap;
  for (int i = 0; i < bn; ++i) { bfmap[order[i].x2] = &bfeats[i]; }
  // Build tree
  typedef TTree<NodeData> Tree;
  Tree tree;
  genTree(tree, order, [](Tree::Node& node, Label r)
          { node.data.label = r; });
  // Generate region classification features
  int n = tree.size();
  int rfd = rfeats.front().dim();
  int bfd = bfeats.front().dim();
  std::vector<std::vector<FVal>> rcfeats(n);
  for (int i = 0; i < n; ++i) {
    Label r = tree[i].data.label;
    rcfeats[i].reserve(rfd + 2 * bfd);
    rfmap.find(r)->second->serialize(rcfeats[i]);
    if (tree[i].isLeaf()) {
      for (int d = 0; d < bfd; ++d)
      { rcfeats[i].push_back(BC_FEAT_NULL_VAL); }
    } else { bfmap.find(r)->second->serialize(rcfeats[i]); }
    if (tree[i].isRoot()) {
      for (int d = 0; d < bfd; ++d)
      { rcfeats[i].push_back(BC_FEAT_NULL_VAL); }
    } else {
      bfmap.find(tree[tree[i].parent].data.label)->second->serialize(
          rcfeats[i]);
    }
  }
  writeData(featFile, rcfeats, " ", "\n", FLT_PREC);
  return true;
}


int main (int argc, char* argv[])
{
  std::vector<std::string>
      _rbImageFiles, _rlImageFiles, _rImageFiles, _bImageFiles;
  std::vector<unsigned int>
      _rbHistBins, _rlHistBins, _rHistBins, _bHistBins;
  std::vector<double>
      _rbHistLowers, _rlHistLowers, _rHistLowers, _bHistLowers;
  std::vector<double>
      _rbHistUppers, _rlHistUppers, _rHistUppers, _bHistUppers;
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("segImage,s", bpo::value<std::string>(&segImageFile)->required(),
       "Input initial segmentation image file name")
      ("mergeOrder,o", bpo::value<std::string>(&mergeOrderFile)->required(),
       "Input merging order file name")
      ("saliency,y", bpo::value<std::string>(&saliencyFile),
       "Input merging saliency file name (optional)")
      ("rbi", bpo::value<std::vector<std::string>>(&_rbImageFiles),
       "Input real image file name(s) (optional)")
      ("rbb", bpo::value<std::vector<unsigned int>>(&_rbHistBins),
       "Input real image histogram bins")
      ("rbl", bpo::value<std::vector<double>>(&_rbHistLowers),
       "Input real image histogram lowers")
      ("rbu", bpo::value<std::vector<double>>(&_rbHistUppers),
       "Input real image histogram uppers")
      ("rli", bpo::value<std::vector<std::string>>(&_rlImageFiles),
       "Input region label image file names(s) (optional)")
      ("rlb", bpo::value<std::vector<unsigned int>>(&_rlHistBins),
       "Input region label image histogram bins")
      ("rll", bpo::value<std::vector<double>>(&_rlHistLowers),
       "Input region label image histogram lowers")
      ("rlu", bpo::value<std::vector<double>>(&_rlHistUppers),
       "Input region label image histogram uppers")
      ("ri", bpo::value<std::vector<std::string>>(&_rImageFiles),
       "Input excl. region image file name(s) (optional)")
      ("rb", bpo::value<std::vector<unsigned int>>(&_rHistBins),
       "Input excl. region image histogram bins")
      ("rl", bpo::value<std::vector<double>>(&_rHistLowers),
       "Input excl. region image histogram lowers")
      ("ru", bpo::value<std::vector<double>>(&_rHistUppers),
       "Input excl. boundary image histogram uppers")
      ("bi", bpo::value<std::vector<std::string>>(&_bImageFiles),
       "Input excl. boundary image file name(s) (optional)")
      ("bb", bpo::value<std::vector<unsigned int>>(&_bHistBins),
       "Input excl. boundary image histogram bins")
      ("bl", bpo::value<std::vector<double>>(&_bHistLowers),
       "Input excl. boundary image histogram lowers")
      ("bu", bpo::value<std::vector<double>>(&_bHistUppers),
       "Input excl. boundary image histogram uppers")
      ("pb", bpo::value<std::string>(&pbImageFile)->required(),
       "Boundary image file for image-based shape features")
      ("maskImage,m", bpo::value<std::string>(&maskImageFile),
       "Input mask image file name")
      ("s0", bpo::value<double>(&initSal),
       "Initial saliency [default: 1.0]")
      ("sb", bpo::value<double>(&salBias),
       "Saliency bias [default: 1.0]")
      ("bt",
       bpo::value<std::vector<double>>(&boundaryThresholds)->multitoken(),
       "Thresholds for image-based shape features (e.g. --bt 0.2 0.5 0.8)")
      ("ns,n", bpo::value<bool>(&normalizeShape),
       "Whether to normalize size and length [default: false]")
      ("logs,l", bpo::value<bool>(&useLogShape),
       "Whether to use logarithms of shape as features [default: false]")
      ("feat,r", bpo::value<std::string>(&featFile)->required(),
       "Output region classification feature file name");
  if (!parse(argc, argv, opts))
  { perr("Error: unable to parse input arguments"); }
  rbImageFiles.reserve(_rbImageFiles.size());
  rlImageFiles.reserve(_rlImageFiles.size());
  rImageFiles.reserve(_rImageFiles.size());
  bImageFiles.reserve(_bImageFiles.size());
  for (int i = 0; i < _rbImageFiles.size(); ++i) {
    rbImageFiles.emplace_back(_rbImageFiles[i], _rbHistBins[i],
                              _rbHistLowers[i], _rbHistUppers[i]);
  }
  for (int i = 0; i < _rlImageFiles.size(); ++i) {
    rlImageFiles.emplace_back(_rlImageFiles[i], _rlHistBins[i],
                              _rlHistLowers[i], _rlHistUppers[i]);
  }
  for (int i = 0; i < _rImageFiles.size(); ++i) {
    rImageFiles.emplace_back(_rImageFiles[i], _rHistBins[i],
                             _rHistLowers[i], _rHistUppers[i]);
  }
  for (int i = 0; i < _bImageFiles.size(); ++i) {
    bImageFiles.emplace_back(_bImageFiles[i], _bHistBins[i],
                             _bHistLowers[i], _bHistUppers[i]);
  }
  return operation() ? EXIT_SUCCESS: EXIT_FAILURE;
}
