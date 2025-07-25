#ifndef version_h
#define version_h

#include "application.h"  // for String
#define xstr(s) str(s)
#define str(s) #s

const String version = "g20250612a";  // deviceOS@5.6.0
// g20250612 is catch functional Vb failure (soft) and revert voc(soc) for BB. 'a' is nom vsat sp
// g20241006 is fix for amp wrap windup limits
// g20240909 is bug fix for noise sensitivity
// g20240902 is initial release of HI_LO sensor configuration
// g20240704 is HI_LO sensor configuration
// g20240331 is garage modifications and two-stage current sensing
// g20240109 is full testing, e.g. allIn
// g20231111b is Talk function streamline
// g20231111a (tab in GitHub) is g20231111 cleaned up for a rogue Talk.h function that was printing to stdout continuously

#endif
