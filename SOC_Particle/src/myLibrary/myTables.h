#ifndef _myTables_h
#define _myTables_h
// #define t_float float

// Interpolating, clipping, 1 and 2-D arbitrarily spaced table look-up

void binsearch(float x, float *v, int n, int *high, int *low, float *dx);
float tab1(float x, float *v, float *y, int n);
float tab1clip(float x, float *v, float *y, int n);
float tab2(float x1, float x2, float *v1, float *v2, float *y, int n1, int n2);

class TableInterp
{
public:
  TableInterp();
  TableInterp(const unsigned int n, float x[]);
  virtual ~TableInterp();
  // operators
  // functions
  virtual float interp(void);
  void pretty_print();
protected:
  unsigned int n1_;
  float *x_;
  float *v_;
};

// 1-D Interpolation Table Lookup
class TableInterp1D : public TableInterp
{
public:
  TableInterp1D();
  TableInterp1D(const unsigned int n, float x[], float v[]);
  ~TableInterp1D();
  //operators
  //functions
  virtual float interp(float x);

protected:
};

// 1-D Interpolation Table Lookup with Clipping
class TableInterp1Dclip : public TableInterp
{
public:
  TableInterp1Dclip();
  TableInterp1Dclip(const unsigned int n, float x[], float v[]);
  ~TableInterp1Dclip();
  //operators
  //functions
  virtual float interp(float x);

protected:
};

// 2-D Interpolation Table Lookup
class TableInterp2D : public TableInterp
{
public:
  TableInterp2D();
  TableInterp2D(const unsigned int n, const unsigned int m, float x[],
                float y[], float v[]);
  ~TableInterp2D();
  //operators
  void put_dx(const float inp) { dx_ = inp; }
  void put_dy(const float inp) { dy_ = inp; }
  void put_dz(const float inp) { dz_ = inp; }

  //functions
  virtual float interp(float x, float y);
  void pretty_print();

protected:
  float dx_;  // Bias on input into table lookup
  float dy_;  // Bias on input into table lookup
  float dz_;  // Bias on calculated output of table lookup
  unsigned int n2_;
  float *y_;
};

#endif

