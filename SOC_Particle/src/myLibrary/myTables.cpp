#include <Arduino.h> //needed for Serial.println
#include "myTables.h"
#include "math.h"
#include "../constants.h"
#include "../parameters.h"
extern SavedPars sp; // Various parameters to be static at system level and saved through power cycle

// Global variables
extern char buffer[256];

// Interpolating, clipping, 1 and 2-D arbitrarily spaced table look-up

/* B I N S E A R C H
*
*   Purpose:    Find x in { v[0] <= v[1] <= ... ,= v[n-1] } and calculate
*               the fraction of range that x is positioned.
*
*   Author:     Dave Gutz 09-Jun-90.
*   Revisions:  Dave Gutz 20-Aug-90 Input x instead of *x to protect
*                   integrity of calling function.
*                      16-Aug-93   Pointers.
*   Inputs:
*       Name        Type        Length      Definition
*       x           double      1           Input to vector.
*       n           int         1           Size of vector.
*       v           double      n           Vector.
*   Outputs:
*       Name        Type        Length      Definition
*       *dx         double      1           Fraction of range for x.
*       *low        int         1           Current low end of range.
*       *high       int         1           Current high end of range.
*   Hardware dependencies:  ANSI C.
*   Header needed in scope of caller:   None.
*   Global variables used:  None.
*   Functions called:   None.
*/
void binsearch(double x, double *v, int n, int *high, int *low, double *dx)
{
  int mid;

  /* Initialize high and low  */
  *low = 0;
  *high = n - 1;

  /* Check endpoints  */
  if (x >= *(v + *high))
  {
    *low = *high;
    *dx = 0.;
  }
  else if (x <= *(v + *low))
  {
    *high = *low;
    *dx = 0.;
  }

  /* Search if necessary  */
  else
  {
    while ((*high - *low) > 1)
    {
      mid = (*low + *high) / 2;
      if (*(v + mid) > x)
        *high = mid;
      else
        *low = mid;
    }
    *dx = (x - *(v + *low)) / (*(v + *high) - *(v + *low));
    if ( sp.debug()==93 )
        Serial.printf("binsearch: x %19.15f high %d low %d v[high] %19.15f v[low] %19.15f dx %19.15f\n", x, *high, *low, *(v + *high), *(v + *low), *dx);
  }
} /* End binsearch    */

/* T A B 1
*
*   Purpose:    Univariant arbitrarily spaced table look-up.
*
*   Author:     Dave Gutz 09-Jun-90.
*   Revisions:  Dave Gutz 20-Aug-90 Input x instead of *x to protect
*                   integrity of calling function.
*                         16-Aug-93   Pointers.
*   Inputs:
*       Name        Type        Length      Definition
*       n           int         1           Number of points.
*       x           double      1           Independent variable.
*       v           double      n           Breakpoint table.
*       y           double      n           Table data.
*   Outputs:
*       Name        Type        Length      Definition
*       tab1        double      1           Result of table lookup.
*   Hardware dependencies:  ANSI C.
*   Header needed in scope of caller:   None.
*   Global variables used:  None.
*   Functions called:   binsearch.
*/
double tab1(double x, double *v, double *y, int n)
{
  double dx;
  int high, low;
  void binsearch(double x, double *v, int n, int *high,
                 int *low, double *dx);
  if (n < 1)
    return y[0];
  binsearch(x, v, n, &high, &low, &dx);
  return *(y + low) + dx * (*(y + high) - *(y + low));
} /* End tab1 */

/* tab1clip:    Univariant arbitrarily spaced table look-up with clipping.
*
*   Author:     Dave Gutz 20-Nov-16
*   Inputs:
*       n           Number of points
*       x           Independent variable
*       v           Breakpoint table
*       y           Table data
*   Outputs:
*       tab1        Result of table lookup
*/
double tab1clip(double x, double *v, double *y, int n)
{
  double dx;
  int high, low;
  void binsearch(double x, double *v, int n, int *high,
                 int *low, double *dx);
  if (n < 1)
    return y[0];
  binsearch(x, v, n, &high, &low, &dx);
  return *(y + low) + fmax(fmin(dx, 1.), 0.) * (*(y + high) - *(y + low));
} /* End tab1clip */

/* T A B 2
*
*   Purpose:    Bivariant arbitrarily spaced table look-up.  Clips
*
*   Author:     Dave Gutz 20-Aug-90.
*   Revisions:            16-Aug-93   Pointers.
*   Inputs:
*       Name        Type        Length      Definition
*       n1          int         1           Number of ind var #1 brkpts.
*       n2          int         1           Number of ind var #2 brkpts.
*       x1          double      1           Independent variable #1.
*       x2          double      1           Independent variable #2.
*       v1          double      n1          Breakpoints for var #1.
*       v2          double      n2          Breakpoints for var #2.
*       y           double      n1*n2       Table data.
*   Outputs:
*       Name        Type        Length      Definition
*       tab2        double      1           Result of table lookup.
*   Hardware dependencies:  ANSI C.
*   Header needed in scope of caller:   None.
*   Global variables used:  None.
*   Functions called:   binsearch (natively clipping)
*/
double tab2(double x1, double x2, double *v1, double *v2, double *y, int n1,
            int n2)
{
  double dx1, dx2, r0, r1;
  int high1, high2, low1, low2, temp1, temp2;
  void binsearch(double x, double *v, int n, int *high,
                 int *low, double *dx);
  if (n1 < 1 || n2 < 1)
    return y[0];
  binsearch(x1, v1, n1, &high1, &low1, &dx1);  // clips
  binsearch(x2, v2, n2, &high2, &low2, &dx2);  // clips
  temp1 = low2 * n1 + low1;
  temp2 = high2 * n1 + low1;
  r0 = *(y + temp1) + dx1 * (*(y + low2 * n1 + high1) - *(y + temp1));
  r1 = *(y + temp2) + dx1 * (*(y + high2 * n1 + high1) - *(y + temp2));
  double result = r0 + dx2 * (r1 - r0);
  if ( sp.debug()==93 )
    Serial.printf("tab2: x %7.3f y %7.3f high1 %d high2 %d low1 %d low2 %d  temp1 %d temp2 %d dx1 %17.15f dx2 %17.15f r0 %17.15f r1 %17.15f result %19.15f\n", \
        x1, x2, high1, high2, low1, low2, temp1, temp2, dx1, dx2, r0, r1, result);
  return result;
} /* End tab2 */

// class TableInterp
// constructors
TableInterp::TableInterp()
    : n1_(0) {}
TableInterp::TableInterp(const uint16_t n, double x[])
    : n1_(n)
{
  x_ = new double[n1_];
  for (uint16_t i = 0; i < n1_; i++)
  {
    x_[i] = x[i];
  }
}

TableInterp::~TableInterp()
{
  delete x_;
}
// operators
// functions
double TableInterp::interp(void)
{
  return (-999.);
}
void TableInterp::pretty_print(void)
{
#ifndef SOFT_DEPLOY_PHOTON
  uint16_t i;
  Serial.printf("    x={");
  for ( i = 0; i < n1_; i++ )
  {
     Serial.printf("%7.3f, ", x_[i]);
  }
  Serial.printf("};\n");
  Serial.printf("    v={");
  for ( i = 0; i < n1_; i++ )
  {
     Serial.printf("%7.3f, ", v_[i]);
  }
  Serial.printf("};\n");
#else
     Serial.printf("TableInterp: silent DEPLOY\n");
#endif
}

// 1-D Interpolation Table Lookup
// constructors
TableInterp1D::TableInterp1D() : TableInterp() {}
TableInterp1D::TableInterp1D(const uint16_t n, double x[], double v[])
    : TableInterp(n, x)
{
  v_ = new double[n1_];
  for (uint16_t i = 0; i < n1_; i++)
  {
    v_[i] = v[i];
  }
}
TableInterp1D::~TableInterp1D()
{
  delete v_;
}
// operators
// functions
double TableInterp1D::interp(double x)
{
  return (tab1(x, x_, v_, n1_));
}

// 1-D Interpolation Table Lookup
// constructors
TableInterp1Dclip::TableInterp1Dclip() : TableInterp() {}
TableInterp1Dclip::TableInterp1Dclip(const uint16_t n, double x[], double v[])
    : TableInterp(n, x)
{
  v_ = new double[n1_];
  for (uint16_t i = 0; i < n1_; i++)
  {
    v_[i] = v[i];
  }
}
TableInterp1Dclip::~TableInterp1Dclip()
{
  delete v_;
}
// operators
// functions
double TableInterp1Dclip::interp(double x)
{
  return (tab1(x, x_, v_, n1_));
}

// 2-D Interpolation Table Lookup
/* Example usage:  see voc_T_ in ../Battery.cpp.
x = {x1, x2, ...xn}
y = {y1, y2, ...ym}
v = {v11, v12, ...v1n, v21, v22, ...v2n, ...............  vm1, vm2, ...vmn}
  = {v11, v12, ...v1n,
     v21, v22, ...v2n,
     ...............
     vm1, vm2, ...vmn}
*/
// constructors
TableInterp2D::TableInterp2D() : TableInterp() {}
TableInterp2D::TableInterp2D(const uint16_t n, const uint16_t m, double x[],
                             double y[], double v[])
    : TableInterp(n, x), dx_(0.), dy_(0.), dz_(0.)
{
  n2_ = m;
  y_ = new double[n2_];
  for (uint16_t j = 0; j < n2_; j++)
  {
    y_[j] = y[j];
  }
  v_ = new double[n1_ * n2_];
  for (uint16_t i = 0; i < n1_; i++)
    for (uint16_t j = 0; j < n2_; j++)
    {
      v_[i + j * n1_] = v[i + j * n1_];
    }
}
TableInterp2D::~TableInterp2D()
{
  delete y_;
  delete v_;
}
// operators
// functions
double TableInterp2D::interp(double x, double y)
{
  return (tab2(x + dx_, y + dy_, x_, y_, v_, n1_, n2_) + dz_);  // clips
}

void TableInterp2D::pretty_print()
{
#ifndef SOFT_DEPLOY_PHOTON
  uint16_t i, j;
  Serial.printf("    dx%7.3f dy%7.3f dz%7.3f\n", dx_, dy_, dz_);
  Serial.printf("    y={"); for ( j=0; j<n2_; j++ ) Serial.printf("%7.3f, ", y_[j] - dy_); Serial.printf("};\n");
  Serial.printf("    x={"); for ( i=0; i<n1_; i++ ) Serial.printf("%7.3f, ", x_[i] - dx_); Serial.printf("};\n");
  Serial.printf("    v={\n");
  for ( j=0; j<n2_; j++ )
  {
    Serial.printf("      {");
    for ( i=0; i<n1_; i++ ) Serial.printf("%7.3f, ", v_[j*n1_+i] + dz_);
    Serial.printf("},\n");
  }
  Serial.printf("      };\n");
#else
     Serial.printf("TableInterp2D: silent DEPLOY\n");
#endif
}
