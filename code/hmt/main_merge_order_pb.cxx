#include "util/image_io.hxx"
#include "util/struct_merge.hxx"
#include "util/text_io.hxx"
#include "util/text_cmd.hxx"
using namespace glia;

std::string segImageFile;
std::string pbImageFile;
std::string maskImageFile;
int type = 1;  // 1 - median, 2 - mean
std::string mergeOrderFile;
std::string saliencyFile;

bool operation ()
{
  std::vector<TTriple<Label>> order;
  std::vector<double> saliencies;
  auto segImage = readImage<LabelImage<DIMENSION>>(segImageFile);
  auto pbImage = readImage<RealImage<DIMENSION>>(pbImageFile);
  auto mask = maskImageFile.empty()?
      LabelImage<DIMENSION>::Pointer(nullptr):
      readImage<LabelImage<DIMENSION>>(maskImageFile);
  typedef TRegionMap<Label, Point<DIMENSION>> RegionMap;
  RegionMap rmap(segImage, mask, true); // Only use contours
  if (type == 1) {
//std::cout << "genMergeOrderGreedyUsingPbApproxMedian" << std::endl;
    auto faux = [](std::vector<double>& data, Label r0, Label r1) { std::cout << "r0=" << r0 << ", r1=" << r1 << "\ndata: " << data << std::endl; };
    auto f_true = [](TBoundaryTable<std::vector<double>, RegionMap>& bt, TBoundaryTable<std::vector<double>, RegionMap>::iterator it) -> bool { std::cout << "Check boundary table condition" << std::endl; return true; };
    genMergeOrderGreedyUsingPbApproxMedian(
      order, saliencies, rmap, false, pbImage, f_true, faux);
  } else if (type == 2) {
    genMergeOrderGreedyUsingPbMean(
        order, saliencies, rmap, false, pbImage, f_true
        <TBoundaryTable<std::pair<double, int>, RegionMap>&,
        TBoundaryTable<std::pair<double, int>, RegionMap>::iterator>);
  } else { perr("Error: unsupported boundary stats type..."); }
  if (!mergeOrderFile.empty()) { writeData(mergeOrderFile, order, "\n"); }
  if (!saliencyFile.empty()) { writeData(saliencyFile, saliencies, "\n"); }
  return true;
}


int main (int argc, char* argv[])
{
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("segImage,s", bpo::value<std::string>(&segImageFile)->required(),
       "Input initial segmentation image file name")
      ("pbImage,p", bpo::value<std::string>(&pbImageFile)->required(),
       "Input boundary probability image file name")
      ("maskImage,m", bpo::value<std::string>(&maskImageFile),
       "Input mask image file name (optional)")
      ("type,t", bpo::value<int>(&type),
       "Boundary intensity stats type (1: median, 2: mean) [default: 1]")
      ("mergeOrder,o", bpo::value<std::string>(&mergeOrderFile),
       "Output merging order file name (optional)")
      ("saliency,y", bpo::value<std::string>(&saliencyFile),
       "Output merging saliency file name (optional)");
  return parse(argc, argv, opts) && operation() ?
      EXIT_SUCCESS : EXIT_FAILURE;
}
