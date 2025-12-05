/* Eigen library by Someone
 */

#include "Eigen.h"

/**
 * Constructor.
 */
Eigen::Eigen()
{
  // be sure not to call anything that requires hardware be initialized here, put those in begin()
}

/**
 * Example method.
 */
void Eigen::begin()
{
    // initialize hardware
    Serial.println("called begin");
}

/**
 * Example method.
 */
void Eigen::process()
{
    // do something useful
    Serial.println("called process");
    doit();
}

/**
* Example private method
*/
void Eigen::doit()
{
    Serial.println("called doit");
}
