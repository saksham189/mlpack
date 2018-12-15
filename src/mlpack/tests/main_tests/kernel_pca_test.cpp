/**
 * @file kernel_pca_test.cpp
 * @author Saksham Bansal
 *
 * Test mlpackMain() of kernel_pca_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "KernelPrincipalComponentsAnalysis";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/kernel_pca/kernel_pca_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct KernelPCATestFixture
{
 public:
  KernelPCATestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~KernelPCATestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

static void ResetSettings()
{
  bindings::tests::CleanMemory();
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);
}

BOOST_FIXTURE_TEST_SUITE(KernelPCAMainTest, KernelPCATestFixture);

/**
 * Make sure that if we ask for a dataset in 3 dimensions back, we get it.
 */
BOOST_AUTO_TEST_CASE(KernelPCADimensionTest)
{
  std::string kernels[] = {
    "linear", "gaussian", "polynomial",
    "hyptan", "laplacian", "epanechnikov", "cosine"
  };

  for(std::string& kernel: kernels)
  {
    ResetSettings();
    arma::mat x = arma::randu<arma::mat>(5, 5);
    // Random input, new dimensionality of 3.
    SetInputParam("input", std::move(x));
    SetInputParam("new_dimensionality", (int) 3);
    SetInputParam("kernel", kernel);
    mlpackMain();

    // Now check that the output has 3 dimensions.
    BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_rows, 3);
    BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols, 5);
  }
}

/**
 * Make sure that centering the dataset makes a difference.
 */
BOOST_AUTO_TEST_CASE(KernelPCACenterTest)
{
  std::string kernels[] = {
    "linear", "gaussian", "polynomial",
    "hyptan", "laplacian", "epanechnikov", "cosine"
  };

  for(std::string& kernel: kernels)
  {
    ResetSettings();

    //Get output without centering the dataset.
    arma::mat x = arma::randu<arma::mat>(5, 5);
    SetInputParam("input", x);
    SetInputParam("new_dimensionality", (int) 3);
    SetInputParam("kernel", kernel);

    mlpackMain();
    arma::mat output1 = CLI::GetParam<arma::mat>("output");

    //Get output after centering the dataset.
    SetInputParam("input", std::move(x));
    SetInputParam("center", true);

    mlpackMain();
    arma::mat output2 = CLI::GetParam<arma::mat>("output");

    // The resulting matrices should be different.
    BOOST_REQUIRE(arma::any(arma::vectorise(output1 != output2)));
  }
}

/**
 * Check that we can't specify an invalid new dimensionality.
 */
BOOST_AUTO_TEST_CASE(KernelPCATooHighNewDimensionalityTest)
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  SetInputParam("input", std::move(x));
  SetInputParam("new_dimensionality", (int) 7); // Invalid.
  SetInputParam("kernel", (std::string)"linear");

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that error is thrown when no kernel is specified.
 */
BOOST_AUTO_TEST_CASE(KernelPCANoKernelTest)
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  SetInputParam("input", std::move(x));
  SetInputParam("new_dimensionality", (int) 3);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that error is thrown if unknown sampling scheme is specified.
 */
BOOST_AUTO_TEST_CASE(KernelPCABadSamplingTest)
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  SetInputParam("input", std::move(x));
  SetInputParam("new_dimensionality", (int) 3);
  SetInputParam("kernel", (std::string)"linear");
  SetInputParam("nystroem_method", true);
  SetInputParam("sampling", (std::string)"bad");

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Test that bandwidth effects the result for gaussian, epanechnikov
 * and laplacian kernels.
 */
BOOST_AUTO_TEST_CASE(KernelPCABandWidthTest)
{

  std::string kernels[] = {
    "gaussian", "epanechnikov", "laplacian"
  };

  for(std::string& kernel: kernels)
  {
    ResetSettings();
    arma::mat x = arma::randu<arma::mat>(5, 5);


    //Get output using bandwidth 1.
    SetInputParam("input", x);
    SetInputParam("new_dimensionality", (int) 3);
    SetInputParam("kernel", (std::string)kernel);
    SetInputParam("bandwidth", (double)1);

    mlpackMain();
    arma::mat output1 = CLI::GetParam<arma::mat>("output");

    //Get output using bandwidth 2.
    SetInputParam("input", std::move(x));
    SetInputParam("bandwidth", (double)2);

    mlpackMain();
    arma::mat output2 = CLI::GetParam<arma::mat>("output");

    // The resulting matrices should be different.
    BOOST_REQUIRE(arma::any(arma::vectorise(output1 != output2)));
  }

}

/**
 * Test that offset effects the result for polynomial and hyptan kernels.
 */
BOOST_AUTO_TEST_CASE(KernelPCAOffsetTest)
{

  std::string kernels[] = {
    "polynomial", "hyptan"
  };

  for(std::string& kernel: kernels)
  {
    ResetSettings();
    arma::mat x = arma::randu<arma::mat>(5, 5);

    SetInputParam("input", x);
    SetInputParam("new_dimensionality", (int) 3);
    SetInputParam("kernel", (std::string)kernel);
    SetInputParam("offset", (double)1);

    mlpackMain();
    arma::mat output1 = CLI::GetParam<arma::mat>("output");

    SetInputParam("input", std::move(x));
    SetInputParam("offset", (double)2);

    mlpackMain();
    arma::mat output2 = CLI::GetParam<arma::mat>("output");

    // The resulting matrices should be different.
    BOOST_REQUIRE(arma::any(arma::vectorise(output1 != output2)));
  }

}

/**
 * Test that degree effects the result for polynomial kernel.
 */
BOOST_AUTO_TEST_CASE(KernelPCADegreeTest)
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  SetInputParam("input", x);
  SetInputParam("new_dimensionality", (int) 3);
  SetInputParam("kernel", (std::string)"polynomial");
  SetInputParam("degree", (double)2);

  mlpackMain();
  arma::mat output1 = CLI::GetParam<arma::mat>("output");

  SetInputParam("input", std::move(x));
  SetInputParam("degree", (double)3);

  mlpackMain();
  arma::mat output2 = CLI::GetParam<arma::mat>("output");

  // The resulting matrices should be different.
  BOOST_REQUIRE(arma::any(arma::vectorise(output1 != output2)));

}

/**
 * Test that kernel scale effects the result for hyptan kernel.
 */
BOOST_AUTO_TEST_CASE(KernelPCAKernelScaleTest)
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  SetInputParam("input", x);
  SetInputParam("new_dimensionality", (int) 3);
  SetInputParam("kernel", (std::string)"hyptan");
  SetInputParam("kernel_scale", (double)2);

  mlpackMain();
  arma::mat output1 = CLI::GetParam<arma::mat>("output");

  SetInputParam("input", std::move(x));
  SetInputParam("kernel_scale", (double)3);

  mlpackMain();
  arma::mat output2 = CLI::GetParam<arma::mat>("output");

  // The resulting matrices should be different.
  BOOST_REQUIRE(arma::any(arma::vectorise(output1 != output2)));

}

/**
 * Test that using a sampling scheme with nystroem method makes a difference.
 */

static void printMat(arma::mat x){
  x.print(std::cout);cout <<endl;
}

BOOST_AUTO_TEST_CASE(KernelPCASamplingSchemeTest)
{
  std::string kernels[] = {
    "gaussian"};

  for(std::string& kernel: kernels)
  {
    ResetSettings();
    arma::mat x = arma::randu<arma::mat>(5, 5);

    SetInputParam("input", x);
    SetInputParam("kernel", (std::string)kernel);
    SetInputParam("nystroem_method", true);
    SetInputParam("new_dimensionality", (int)1);
    SetInputParam("sampling", (std::string)"kmeans");
    mlpackMain();
    const arma::mat output1 = CLI::GetParam<arma::mat>("output");
    printMat(output1);

    ResetSettings();

    SetInputParam("input", x);
    SetInputParam("kernel", (std::string)kernel);
    SetInputParam("nystroem_method", true);
    SetInputParam("new_dimensionality", (int)1);
    SetInputParam("sampling", (std::string)"random");
    mlpackMain();
    const arma::mat output2 = CLI::GetParam<arma::mat>("output");
    printMat(output2);

    ResetSettings();

    SetInputParam("input", x);
    SetInputParam("kernel", (std::string)kernel);
    SetInputParam("nystroem_method", true);
    SetInputParam("new_dimensionality", (int)1);
    SetInputParam("sampling", (std::string)"ordered");
    mlpackMain();
    const arma::mat output3 = CLI::GetParam<arma::mat>("output");
    printMat(output3);

    output1.print(std::cout);
    std::cout <<endl;
    output2.print(std::cout);
    std::cout <<endl;
    output3.print(std::cout);
    std::cout <<endl;
    // The resulting matrices should be different.
    BOOST_REQUIRE(arma::any(arma::vectorise(output1 != output2)));
    BOOST_REQUIRE(arma::any(arma::vectorise(output2 != output3)));
    BOOST_REQUIRE(arma::any(arma::vectorise(output1 != output3)));
  }
}


BOOST_AUTO_TEST_SUITE_END();
