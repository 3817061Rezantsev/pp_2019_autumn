// Copyright 2019 Rezantsev Sergey
#include <gtest/gtest.h>
#include <math.h>
#include <gtest-mpi-listener.hpp>
#include <vector>
#include "../../../modules/task_2/rezantsev_s_hor_gauss/hor_gauss.h"

TEST(Hor_Gauss_MPI, gauss_test_on_matrix_3x4) {
  std::vector<double> a(12);
  int rank;
  a[0] = 3;
  a[1] = 4;
  a[2] = 8;
  a[3] = 1;
  a[4] = 2;
  a[5] = 5;
  a[6] = 6;
  a[7] = 3;
  a[8] = 1;
  a[9] = 3;
  a[10] = 7;
  a[11] = 9;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<double> res(3);
  res = getGauss(a, 3);
  bool check = isItTrueAnswer(a, 3, res);
  if (rank == 0) {
    EXPECT_EQ(check, true);
  }
}

TEST(Hor_Gauss_MPI, pargauss_test_on_matrix_3x4) {
  std::vector<double> a(12);
  int rank;
  a[0] = 3;
  a[1] = 4;
  a[2] = 8;
  a[3] = 1;
  a[4] = 2;
  a[5] = 5;
  a[6] = 6;
  a[7] = 3;
  a[8] = 1;
  a[9] = 3;
  a[10] = 7;
  a[11] = 9;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<double> res = getParGauss(a, 3);
  if (rank == 0) {
    bool c = isItTrueAnswer(a, 3, res);
    EXPECT_EQ(c, true);
  }
}

TEST(Hor_Gauss_MPI, effective_test_on_matrix_30x31) {
  std::vector<double> a = getRandMatrix(30);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // double startPar = MPI_Wtime();
  std::vector<double> a1 = getParGauss(a, 30);
  // double endPar = MPI_Wtime();
  if (rank == 0) {
    // double startSeq = MPI_Wtime();
    std::vector<double> a2 = getGauss(a, 30);
    // double endSeq = MPI_Wtime();
    bool c = isItTrueAnswer(a, 30, a1);
    EXPECT_EQ(c, true);
    // std::cout << "Time seq: " << endSeq - startSeq << std::endl;
    // std::cout << "Time par: " << endPar - startPar << std::endl;
  }
}

TEST(Hor_Gauss_MPI, pargauss_test_on_matrix_2x3) {
  std::vector<double> a = getRandMatrix(2);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<double> res(2);
  res = getParGauss(a, 2);
  if (rank == 0) {
    bool check = isItTrueAnswer(a, 2, res);
    EXPECT_EQ(check, true);
  }
}

TEST(Hor_Gauss_MPI, pargauss_test_on_matrix_15x16) {
  std::vector<double> a = getRandMatrix(15);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<double> res(15);
  res = getParGauss(a, 15);
  if (rank == 0) {
    bool check = isItTrueAnswer(a, 15, res);
    EXPECT_EQ(check, true);
  }
}

TEST(Hor_Gauss_MPI, pargauss_test_on_matrix_11x12) {
  std::vector<double> a = getRandMatrix(11);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<double> res(11);
  res = getParGauss(a, 11);
  if (rank == 0) {
    bool check = isItTrueAnswer(a, 11, res);
    EXPECT_EQ(check, true);
  }
}

TEST(Hor_Gauss_MPI, pargauss_test_on_matrix_10x11) {
  std::vector<double> a = getRandMatrix(10);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<double> res(10);
  res = getParGauss(a, 10);
  if (rank == 0) {
    bool check = isItTrueAnswer(a, 10, res);
    EXPECT_EQ(check, true);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
  ::testing::TestEventListeners& listeners =
      ::testing::UnitTest::GetInstance()->listeners();

  listeners.Release(listeners.default_result_printer());
  listeners.Release(listeners.default_xml_generator());

  listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
  return RUN_ALL_TESTS();
}
