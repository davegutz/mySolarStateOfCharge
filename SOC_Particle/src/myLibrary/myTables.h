#pragma once
// #define t_double double

// Interpolating, clipping, 1 and 2-D arbitrarily spaced table look-up

void binsearch(double x, double *v, int n, int *high, int *low, double *dx);
double tab1(double x, double *v, double *y, int n);
double tab1clip(double x, double *v, double *y, int n);
double tab2(double x1, double x2, double *v1, double *v2, double *y, int n1, int n2);

class TableInterp
{
public:
  TableInterp();
  TableInterp(const uint16_t n, double x[]);
  virtual ~TableInterp();
  // operators
  // functions
  virtual double interp(void);
  void pretty_print();
protected:
  uint16_t n1_;
  double *x_;
  double *v_;
};

// 1-D Interpolation Table Lookup
class TableInterp1D : public TableInterp
{
public:
  TableInterp1D();
  TableInterp1D(const uint16_t n, double x[], double v[]);
  ~TableInterp1D();
  //operators
  //functions
  virtual double interp(double x);

protected:
};

// 1-D Interpolation Table Lookup with Clipping
class TableInterp1Dclip : public TableInterp
{
public:
  TableInterp1Dclip();
  TableInterp1Dclip(const uint16_t n, double x[], double v[]);
  ~TableInterp1Dclip();
  //operators
  //functions
  virtual double interp(double x);

protected:
};

// 2-D Interpolation Table Lookup
class TableInterp2D : public TableInterp
{
public:
  TableInterp2D();
  TableInterp2D(const uint16_t n, const uint16_t m, double x[],
                double y[], double v[]);
  ~TableInterp2D();
  //operators
  void put_dx(const double inp) { dx_ = inp; }
  void put_dy(const double inp) { dy_ = inp; }
  void put_dz(const double inp) { dz_ = inp; }

  //functions
  virtual double interp(double x, double y);
  void pretty_print();

protected:
  double dx_;  // Bias on input into table lookup
  double dy_;  // Bias on input into table lookup
  double dz_;  // Bias on calculated output of table lookup
  uint16_t n2_;
  double *y_;
};

