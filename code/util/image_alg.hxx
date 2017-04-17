#ifndef _glia_util_image_alg_hxx_
#define _glia_util_image_alg_hxx_

#include "util/image.hxx"
#include "itkMorphologicalWatershedImageFilter.h"
#include "itkWatershedImageFilter.h"
#include "itkCastImageFilter.h"

namespace glia {

template <typename TImageOut, typename TImagePtrIn>
typename TImageOut::Pointer
watershed (TImagePtrIn const& inputImage, double level)
{
  const UInt D = TImage<TImagePtrIn>::ImageDimension;
  auto ws = itk::MorphologicalWatershedImageFilter
      <TImage<TImagePtrIn>, TImageOut>::New();
  ws->SetInput(inputImage);
  ws->SetLevel(level);
  ws->MarkWatershedLineOff();
//  ws->FullyConnectedOn();
  ws->FullyConnectedOff();
  ws->Update();
  return ws->GetOutput();
}

template <typename TImageOut, typename TImagePtrIn>
typename TImageOut::Pointer
watershed_classic (TImagePtrIn const& inputImage, double level, double threshold)
{
  const UInt D = TImage<TImagePtrIn>::ImageDimension;
  typedef itk::WatershedImageFilter< TImage<TImagePtrIn> > WatershedFilterType;
  typedef itk::CastImageFilter< ULongImage<2>, TImageOut > CastFilterType;
  auto ws = WatershedFilterType::New();
  ws->SetInput(inputImage);
  ws->SetLevel(level);
  ws->SetThreshold(threshold);
  ws->Update();
  auto cf = CastFilterType::New();
  cf->SetInput(ws->GetOutput());
  cf->Update();
  return cf->GetOutput();
}

};

#endif
