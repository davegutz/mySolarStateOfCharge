// Example usage for Eigen library by Someone.

#include "Eigen.h"

// Initialize objects from the lib
Eigen eigen;

void setup() {
    // Call functions on initialized library objects that require hardware
    eigen.begin();
}

void loop() {
    // Use the library's initialized objects and functions
    eigen.process();
}
