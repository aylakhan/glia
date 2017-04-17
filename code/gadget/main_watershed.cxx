#include "util/image_alg.hxx"
#include "util/image_io.hxx"
#include "util/text_cmd.hxx"

using namespace glia;

bool operation (std::string const& outputImageFile, std::string const& testOutputImageFile,
                std::string const& inputImageFile,
                double level, double threshold, bool relabel, bool write16, bool compress)
{
  auto inputImage = readImage<RealImage<DIMENSION>>(inputImageFile);
  auto outputImage = watershed<LabelImage<DIMENSION>>(inputImage, level);
  //auto outputImage = watershed_classic<LabelImage<DIMENSION>>(inputImage, level, threshold);
  if (relabel) { relabelImage(outputImage, 0); }
  if (write16) {
    castWriteImage<UInt16Image<DIMENSION>>
        (outputImageFile, outputImage, compress);
  }
  else { writeImage(outputImageFile, outputImage, compress); }
  // test
  castWriteImage<UCharImage<DIMENSION>>
  (testOutputImageFile, outputImage, compress);
  return true;
}


int main (int argc, char* argv[])
{
  std::string inputImageFile, outputImageFile, testOutputImageFile;
  double level, threshold;
  bool relabel = false, write16 = false, compress = false;
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("inputImage,i", bpo::value<std::string>(&inputImageFile)->required(),
       "Input image file name")
      ("level,l", bpo::value<double>(&level)->required(),
       "Watershed water level")
      ("threshold,t", bpo::value<double>(&threshold)->required(),
       "Watershed threshold level")
      ("relabel,r", bpo::value<bool>(&relabel),
       "Whether to relabel output image [default: false]")
      ("write16,u", bpo::value<bool>(&write16),
       "Whether to write to uint16 image [default: false]")
      ("compress,z", bpo::value<bool>(&compress),
       "Whether to compress output image file(s) [default: false]")
      ("outputImage,o",
       bpo::value<std::string>(&outputImageFile)->required(),
       "Output image file name")
      ("toi",
        bpo::value<std::string>(&testOutputImageFile)->required(),
       "Test output image file name");
  return
      parse(argc, argv, opts) &&
      operation(outputImageFile, testOutputImageFile, inputImageFile, level, threshold, relabel, write16,
                compress)? EXIT_SUCCESS: EXIT_FAILURE;
}
