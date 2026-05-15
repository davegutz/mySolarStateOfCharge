//
// MIT License
//
// Copyright (C) 2026 - Dave Gutz
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "hardware/SerialRAM.h"
#include "PrinterPars.h"
#include "ble.h"

void sendTxBuf(const String& txBuf, const bool sendSerial, const bool sendBLE);

extern PrinterPars pr;  // Print buffer

#undef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#undef min
#define min(a, b) (((a) < (b)) ? (a) : (b))

typedef void (*fptr)();

/* Using pointers in building class so all that stuff does not get saved by 'retained' keyword in SOC_Particle.ino.
    Only the *_z parameters at the bottom of Parameters.h are stored in SRAM.  Class initialized by *init in arg list.
*/
class Variable
{
public:
    Variable(){}

    Variable(const String &prefix, const String &code, SerialRAM *ram, const String &description, const String &units,
        const bool check_for_off_on_init=true)
    {
        prefix_ = prefix;
        code_ = code;
        description_ = description.substring(0, 20);
        units_ = units.substring(0, 10);
        check_for_off_on_init_ = check_for_off_on_init;
        is_eeram_ = !(ram==NULL);
        rP_ = ram;
        // app_ = app;
    }

    ~Variable(){}

    fptr app_;
    fptr app() { return app_; }
    // void app(fptr ptr) { app_ = ptr; }
    String code() { return code_; }
    const char* description() { return description_.c_str(); }
    bool success() { return success_; }
    const char* units() { return units_.c_str(); }

    // Placeholders
    virtual void get(){};
    virtual bool is_corrupt(){return false;};
    virtual bool is_eeram(){return is_eeram_;};
    virtual bool is_off(){return false;};
    virtual bool off_nominal(){return false;};
    virtual void print(){};
    virtual bool print_adjust(const String &str){return false;};
    bool print_nominalize()
    {
        set_nominal();
        sendTxBuf(" Nominalizing:  ", true, IN_SERVICE);
        return ( true );
    }
    virtual void set_nominal(){};

protected:
    String code_;
    SerialRAM *rP_;
    address16b addr_;
    String units_;
    String description_;
    bool is_eeram_;      // eeram
    bool check_for_off_on_init_;  // check for off-nominal on initialization
    String prefix_;         // either "* " saved or "  " not saved
    bool success_;       // result of print_adjust
};


class BooleanV: public Variable
{
public:
    BooleanV(){}

    BooleanV(const String &prefix, const String &code, SerialRAM *ram, const String &description, const String &units, const bool min, const bool max,
    bool *store, const bool _default=false, const bool check_for_off_on_init=true):
        Variable(prefix, code, ram, description, units, check_for_off_on_init)
    {
        min_ = min;
        max_ = max;
        val_ = store;
        default_ = max(min(_default, max_), min_);
        if ( check_for_off_on_init_ ) check_set_put(*val_);
    }

    ~BooleanV(){}

    bool check_set_put(bool val)
    {
        if ( val>max_ || val<min_ )
        {
            Serial.printf("%s %s set:: out range %d (%d, %d)\n", code_.c_str(), description_.c_str(), val, min_, max_);
            return false;
        }
        else
        {
            *val_ = val;
            if ( is_eeram_ ) rP_->write(addr_.a16, *val_);
            return true;
        }
    }

    virtual void get()
    {
        if ( is_eeram_ ) *val_ = rP_->read(addr_.a16);
    }

    virtual bool is_corrupt()
    {
        bool corrupt = *val_ > max_ || *val_ < min_;
        if ( corrupt ) sendTxBuf(String::format("\n%s %s corrupt", code_.c_str(), description_.c_str()), true, IN_SERVICE); 
        return corrupt;
    }

    virtual bool is_off()
    {
        return off_nominal() && check_for_off_on_init_;
    }

    virtual bool off_nominal()
    {
        return *val_ != default_;
    }

    void print_str()
    {
        sprintf(pr.buff, " %-20s %9d -> %9d, %10s (%s%-2s)", description_.c_str(), default_, *val_, units_.c_str(), prefix_.c_str(), code_.c_str());
    }

    void print()
    {
        print_str();
        sendTxBuf(String::format("%s\n", pr.buff), true, IN_SERVICE);
    }
    
    void print_help_str()
    {
        sprintf(pr.buff, "%s%-2s= %6d: (%-6d-%6d) [%6d] %s, %s", prefix_.c_str(), code_.c_str(), *val_, min_, max_, default_, description_.c_str(), units_.c_str());
    }

    void print_help()
    {
        print_help_str();
        sendTxBuf(String::format("%s\n", pr.buff), true, IN_SERVICE);
    }

    virtual bool print_adjust(const String &str)
    {
        print();
        if ( str == "" ) success_ = print_nominalize();
        else success_ = check_set_put(str.toInt());
        print();
        return success_;
    }

    bool print_adj_print(const bool input)
    {
        print();
        success_ = check_set_put(input);
        print();
        return success_;
    }
   
    virtual void set_nominal()
    {
        *val_ = default_;
        if ( is_eeram_ ) rP_->write(addr_.a16, *val_);
    }

protected:
    bool *val_;
    bool min_;
    bool max_;
    bool default_;
};


class DoubleV: public Variable
{
public:
    DoubleV(){}

    DoubleV(const String &prefix, const String &code, SerialRAM *ram, const String &description, const String &units, const double min, const double max, double *store,
        const double _default=0, const bool check_for_off_on_init=true):
        Variable(prefix, code, ram, description, units, check_for_off_on_init)
    {
        min_ = min;
        max_ = max;
        val_ = store;
        default_ = max(min(_default, max_), min_);
        if ( check_for_off_on_init_ ) check_set_put(*val_);
    }

    ~DoubleV(){}

    bool check_set_put(double val)
    {
        if ( val>max_ || val<min_ )
        {
            if ( val>max_ || val<min_ ) Serial.printf("%s %s set:: out range %7.3f (%7.3f, %7.3f)\n", code_.c_str(), description_.c_str(), val, min_, max_);
            return false;
        }
        else
        {
            *val_ = val;
            if ( is_eeram_ ) rP_->put(addr_.a16, *val_);
            return true;
        }
    }

    virtual void get()
    {
        if ( is_eeram_ )
        {
            rP_->get(addr_.a16, *val_);
        }
    }

    virtual bool is_corrupt()
    {
        bool corrupt = *val_ > max_ || *val_ < min_;
        if ( corrupt ) sendTxBuf(String::format("\n%s %s corrupt", code_.c_str(), description_.c_str()), true, IN_SERVICE); 
        return corrupt;
    }

    virtual bool is_off()
    {
        return off_nominal() && check_for_off_on_init_;
    }

    virtual bool off_nominal()
    {
        return abs(*val_-default_)>1e-4;
    }

    void print_str()
    {
        sprintf(pr.buff, " %-20s %9.1f -> %9.1f, %10s (%s%-2s)", description_.c_str(), default_, *val_, units_.c_str(), prefix_.c_str(), code_.c_str());
    }
    
    void print()
    {
        print_str();
        sendTxBuf(String::format("%s\n", pr.buff), true, IN_SERVICE);
    }

    void print_help_str()
    {
        sprintf(pr.buff, "%s%-2s= %6.1f: (%-6.1f-%6.1f) [%6.1f] %s, %s", prefix_.c_str(), code_.c_str(), *val_, min_, max_, default_, description_.c_str(), units_.c_str());
    }
    
    void print_help()
    {
        print_help_str();
        sendTxBuf(String::format("%s\n", pr.buff), true, IN_SERVICE);
    }

    virtual bool print_adjust(const String &str)
    {
        print();
        if ( str == "" ) success_ = print_nominalize();
        else success_ = check_set_put(str.toFloat());
        print();
        return success_;
    }

    bool print_adj_print(const double input)
    {
        print();
        success_ = check_set_put(input);
        print();
        return success_;
    }

    virtual void set_nominal()
    {
        *val_ = default_;
        if ( is_eeram_ ) rP_->put(addr_.a16, *val_);
    }

protected:
    double *val_;
    double default_;
    double min_;
    double max_;
};


class FloatV: public Variable
{
public:
    FloatV(){}

    FloatV(const String &prefix, const String &code, SerialRAM *ram, const String &description, const String &units, const float min, const float max,
    float *store, const float _default=0, const bool check_for_off_on_init=true):
        Variable(prefix, code, ram, description, units, check_for_off_on_init)
    {
        min_ = min;
        max_ = max;
        val_ = store;
        default_ = max(min(_default, max_), min_);
        if ( check_for_off_on_init_ ) check_set_put(*val_);
    }

    ~FloatV(){}

    bool check_set_put(float val)
    {
        if ( val>max_ || val<min_ )
        {
            if ( val>max_ || val<min_ ) Serial.printf("%s %s set:: out range %7.3f (%7.3f, %7.3f)\n", code_.c_str(), description_.c_str(), val, min_, max_);
            return false;
        }
        else
        {
            *val_ = val;
            if ( is_eeram_ ) rP_->put(addr_.a16, *val_);
            return true;
        }
    }

    virtual void get()
    {
        if ( is_eeram_ )
        {
            rP_->get(addr_.a16, *val_);
        }
    }

    virtual bool is_corrupt()
    {
        bool corrupt = *val_ > max_ || *val_ < min_;
        if ( corrupt ) sendTxBuf(String::format("\n%s %s corrupt", code_.c_str(), description_.c_str()), true, IN_SERVICE); 
        return corrupt;
    }

    virtual bool is_off()
    {
        return off_nominal() && check_for_off_on_init_;
    }

    virtual bool off_nominal()
    {
        return abs(*val_-default_)>1e-4;
    }

    void print_str()
    {
        sprintf(pr.buff, " %-20s %9.3g -> %9.3g, %10s (%s%-2s)", description_.c_str(), default_, *val_, units_.c_str(), prefix_.c_str(), code_.c_str());
    }

    void print()
    {
        print_str();
        sendTxBuf(String::format("%s\n", pr.buff), true, IN_SERVICE);
    }

    void print_help_str()
    {
        sprintf(pr.buff, "%s%-2s= %6.3g: (%-6.3g-%6.3g) [%6.3g] %s, %s", prefix_.c_str(), code_.c_str(), *val_, min_, max_, default_, description_.c_str(), units_.c_str());
    }

    void print_help()
    {
        print_help_str();
        sendTxBuf(String::format("%s\n", pr.buff), true, IN_SERVICE);
    }

    virtual bool print_adjust(const String &str)
    {
        print();
        if ( str == "" ) success_ = print_nominalize();
        else success_ = check_set_put(str.toFloat());
        print();
        return success_;
    }

    bool print_adj_print(const float input)
    {
        print();
        success_ = check_set_put(input);
        print();
        return success_;
    }

    virtual void set_nominal()
    {
        *val_ = default_;
        if ( is_eeram_ ) rP_->put(addr_.a16, *val_);
    }

protected:
    float *val_;
    float default_;
    float min_;
    float max_;
};


class IntV: public Variable
{
public:
    IntV(){}

    IntV(const String &prefix, const String &code, SerialRAM *ram, const String &description, const String &units, const int min, const int max, int *store,
        const int _default=0, const bool check_for_off_on_init=true):
        Variable(prefix, code, ram, description, units, check_for_off_on_init)
    {
        min_ = min;
        max_ = max;
        val_ = store;
        default_ = max(min(_default, max_), min_);
        if ( check_for_off_on_init_ ) check_set_put(*val_);
    }

    ~IntV(){}

    bool check_set_put(int val)
    {
        if ( val>max_ || val<min_ )
        {
            if ( val>max_ || val<min_ ) Serial.printf("%s %s set:: out range %d (%d, %d)\n", code_.c_str(), description_.c_str(), val, min_, max_);
            return false;
        }
        else
        {
            *val_ = val;
            if ( is_eeram_ ) rP_->put(addr_.a16, *val_);
            return true;
        }
    }

    virtual void get()
    {
        if ( is_eeram_ )
        {
            rP_->get(addr_.a16, *val_);
        }
    }

    virtual bool is_corrupt()
    {
        bool corrupt = *val_ > max_ || *val_ < min_;
        if ( corrupt ) sendTxBuf(String::format("\n%s %s corrupt", code_.c_str(), description_.c_str()), true, IN_SERVICE); 
        return corrupt;
    }

    virtual bool is_off()
    {
        return off_nominal() && check_for_off_on_init_;
    }

    virtual bool off_nominal()
    {
        return *val_ != default_;
    }

    void print_str()
    {
        sprintf(pr.buff, " %-20s %9d -> %9d, %10s (%s%-2s)", description_.c_str(), default_, *val_, units_.c_str(), prefix_.c_str(), code_.c_str());
    }
    void print()
    {
        print_str();
        sendTxBuf(String::format("%s\n", pr.buff), true, IN_SERVICE);
    }
    
    void print_help_str()
    {
      sprintf(pr.buff, "%s%-2s= %6d: (%-6d-%6d) [%6d] %s, %s", prefix_.c_str(), code_.c_str(), *val_, min_, max_, default_, description_.c_str(), units_.c_str());
    }

    void print_help()
    {
        print_help_str();
        sendTxBuf(String::format("%s\n", pr.buff), true, IN_SERVICE);
    }
    
    virtual bool print_adjust(const String &str)
    {
        print();
        if ( str == "" ) success_ = print_nominalize();
        else success_ = check_set_put(str.toInt());
        print();
        return success_;
    }

    bool print_adj_print(const int input)
    {
        print();
        success_ = check_set_put(input);
        print();
        return success_;
    }

    virtual void set_nominal()
    {
        *val_ = default_;
        if ( is_eeram_ ) rP_->put(addr_.a16, *val_);
    }

protected:
    int *val_;
    int min_;
    int max_;
    int default_;
};


class Int8tV: public Variable
{
public:
    Int8tV(){}

    Int8tV(const String &prefix, const String &code, SerialRAM *ram, const String &description, const String &units, const int8_t min, const int8_t max,
        int8_t *store, const int8_t _default=0, const bool check_for_off_on_init=true):
        Variable(prefix, code, ram, description, units, check_for_off_on_init)
    {
        min_ = min;
        max_ = max;
        val_ = store;
        default_ = max(min(_default, max_), min_);
        if ( check_for_off_on_init_ ) check_set_put(*val_);
    }

    ~Int8tV(){}

    bool check_set_put(int8_t val)
    {
        if ( val>max_ || val<min_ )
        {
            if ( val>max_ || val<min_ ) Serial.printf("%s %s set:: out range %d (%d, %d)\n", code_.c_str(), description_.c_str(), val, min_, max_);
            return false;
        }
        else
        {
            *val_ = val;
            if ( is_eeram_ ) rP_->put(addr_.a16, *val_);
            return true;
        }
    }

    virtual void get()
    {
        if ( is_eeram_ )
        {
            rP_->get(addr_.a16, *val_);
        }
    }

    virtual bool is_corrupt()
    {
        bool corrupt = *val_ > max_ || *val_ < min_;
        if ( corrupt ) sendTxBuf(String::format("\n%s %s corrupt", code_.c_str(), description_.c_str()), true, IN_SERVICE); 
        return corrupt;
    }

    virtual bool is_off()
    {
        return off_nominal() && check_for_off_on_init_;
    }

    virtual bool off_nominal()
    {
        return *val_ != default_;
    }

    void print_str()
    {
        sprintf(pr.buff, " %-20s %9d -> %9d, %10s (%s%-2s)", description_.c_str(), default_, *val_, units_.c_str(), prefix_.c_str(), code_.c_str());
    }

    void print()
    {
        print_str();
        sendTxBuf(String::format("%s\n", pr.buff), true, IN_SERVICE);
    }
    
    void print_help_str()
    {
      sprintf(pr.buff, "%s%-2s= %6d: (%-6d-%6d) [%6d] %s, %s", prefix_.c_str(), code_.c_str(), *val_, min_, max_, default_, description_.c_str(), units_.c_str());
    }

    void print_help()
    {
        print_help_str();
        sendTxBuf(String::format("%s\n", pr.buff), true, IN_SERVICE);
    }
    
    virtual bool print_adjust(const String &str)
    {
        print();
        if ( str == "" ) success_ = print_nominalize();
        else success_ = check_set_put(str.toInt());
        print();
        return success_;
    }

    bool print_adj_print(const int8_t input)
    {
        print();
        success_ = check_set_put(input);
        print();
        return success_;
    }

    virtual void set_nominal()
    {
        *val_ = default_;
        if ( is_eeram_ ) rP_->put(addr_.a16, *val_);
    }

protected:
    int8_t *val_;
    int8_t min_;
    int8_t max_;
    int8_t default_;
};


class Uint16tV: public Variable
{
public:
    Uint16tV(){}

    Uint16tV(const String &prefix, const String &code, SerialRAM *ram, const String &description, const String &units, const uint16_t min, const uint16_t max,
    uint16_t *store, const uint16_t _default=0, const bool check_for_off_on_init=true):
        Variable(prefix, code, ram, description, units, check_for_off_on_init)
    {
        min_ = min;
        max_ = max;
        val_ = store;
        default_ = max(min(_default, max_), min_);
        if ( check_for_off_on_init_ ) check_set_put(*val_);
     }

    ~Uint16tV(){}

    bool check_set_put(uint16_t val)
    {
        if ( val>max_ || val<min_ )
        {
            if ( val>max_ || val<min_ ) Serial.printf("%s %s set:: out range %d (%d, %d)\n", code_.c_str(), description_.c_str(), val, min_, max_);
            return false;
        }
        else
        {
            *val_ = val;
            if ( is_eeram_ ) rP_->write(addr_.a16, *val_);
            return true;
        }
    }

    virtual void get()
    {
        if ( is_eeram_ ) *val_ = rP_->read(addr_.a16);
    }

    virtual bool is_corrupt()
    {
        bool corrupt = *val_ > max_ || *val_ < min_;
        if ( corrupt ) sendTxBuf(String::format("\n%s %s corrupt", code_.c_str(), description_.c_str()), true, IN_SERVICE); 
        return corrupt;
    }

    virtual bool is_off()
    {
        return off_nominal() && check_for_off_on_init_;
    }

    void new_maximum(const uint16_t new_max) { max_ = new_max; }
    void new_default(const uint16_t new_def) { default_ = max(min(new_def, max_), min_); }

    virtual bool off_nominal()
    {
        return *val_ != default_;
    }

    void print_str()
    {
        sprintf(pr.buff, " %-20s %9d -> %9d, %10s (%s%-2s)", description_.c_str(), default_, *val_, units_.c_str(), prefix_.c_str(), code_.c_str());
    }

    void print()
    {
        print_str();
        sendTxBuf(String::format("%s\n", pr.buff), true, IN_SERVICE);
    }
    
    void print_help_str()
    {
        sprintf(pr.buff, "%s%-2s= %6d: (%-6d-%6d) [%6d] %s, %s", prefix_.c_str(), code_.c_str(), *val_, min_, max_, default_, description_.c_str(), units_.c_str());
    }

    void print_help()
    {
        print_help_str();
        sendTxBuf(String::format("%s\n", pr.buff), true, IN_SERVICE);
    }

    virtual bool print_adjust(const String &str)
    {
        print();
        if ( str == "" ) success_ = print_nominalize();
        else success_ = check_set_put(str.toInt());
        print();
        return success_;
    }

    bool print_adj_print(const uint16_t input)
    {
        print();
        success_ = check_set_put(input);
        print();
        return success_;
    }
   
    virtual void set_nominal()
    {
        *val_ = default_;
        if ( is_eeram_ ) rP_->write(addr_.a16, *val_);
    }

protected:
    uint16_t *val_;
    uint16_t min_;
    uint16_t max_;
    uint16_t default_;
};

class Uint8tV: public Variable
{
public:
    Uint8tV(){}

    Uint8tV(const String &prefix, const String &code, SerialRAM *ram, const String &description, const String &units, const uint8_t min, const uint8_t max,
    uint8_t *store, const uint8_t _default=0, const bool check_for_off_on_init=true):
        Variable(prefix, code, ram, description, units, check_for_off_on_init)
    {
        min_ = min;
        max_ = max;
        val_ = store;
        default_ = max(min(_default, max_), min_);
        if ( check_for_off_on_init_ ) check_set_put(*val_);
    }

    ~Uint8tV(){}

    bool check_set_put(uint8_t val)
    {
        if ( val>max_ || val<min_ )
        {
            if ( val>max_ || val<min_ ) Serial.printf("%s %s set:: out range %d (%d, %d)\n", code_.c_str(), description_.c_str(), val, min_, max_);
            return false;
        }
        else
        {
            *val_ = val;
            if ( is_eeram_ ) rP_->write(addr_.a16, *val_);
            return true;
        }
    }

    virtual void get()
    {
        if ( is_eeram_ ) *val_ = rP_->read(addr_.a16);
    }

    virtual bool is_corrupt()
    {
        bool corrupt = *val_ > max_ || *val_ < min_;
        if ( corrupt ) sendTxBuf(String::format("\n%s %s corrupt", code_.c_str(), description_.c_str()), true, IN_SERVICE); 
        return corrupt;
    }

    virtual bool is_off()
    {
        return off_nominal() && check_for_off_on_init_;
    }

    virtual bool off_nominal()
    {
        return *val_ != default_;
    }

    void print_str()
    {
        sprintf(pr.buff, " %-20s %9d -> %9d, %10s (%s%-2s)", description_.c_str(), default_, *val_, units_.c_str(), prefix_.c_str(), code_.c_str());
    }

    void print()
    {
        print_str();
        sendTxBuf(String::format("%s\n", pr.buff), true, IN_SERVICE);
    }
    
    void print_help_str()
    {
        sprintf(pr.buff, "%s%-2s= %6d: (%-6d-%6d) [%6d] %s, %s", prefix_.c_str(), code_.c_str(), *val_, min_, max_, default_, description_.c_str(), units_.c_str());
    }

    void print_help()
    {
        print_help_str();
        sendTxBuf(String::format("%s\n", pr.buff), true, IN_SERVICE);
    }

    virtual bool print_adjust(const String &str)
    {
        print();
        if ( str == "" ) success_ = print_nominalize();
        else success_ = check_set_put(str.toInt());
        print();
        return success_;
    }

    bool print_adj_print(const uint8_t input)
    {
        print();
        success_ = check_set_put(input);
        print();
        return success_;
    }
   
    virtual void set_nominal()
    {
        *val_ = default_;
        if ( is_eeram_ ) rP_->write(addr_.a16, *val_);
    }

protected:
    uint8_t *val_;
    uint8_t min_;
    uint8_t max_;
    uint8_t default_;
};


class ULongV: public Variable
{
public:
    ULongV(){}

    ULongV(const String &prefix, const String &code, SerialRAM *ram, const String &description, const String &units, const uint32_t min, const uint32_t max,
    uint32_t *store, const uint32_t _default=0, const bool check_for_off_on_init=true):
        Variable(prefix, code, ram, description, units, check_for_off_on_init)
    {
        min_ = min;
        max_ = max;
        val_ = store;
        default_ = max(min(_default, max_), min_);
        if ( check_for_off_on_init_ ) check_set_put(*val_);
    }

    ~ULongV(){}

    bool check_set_put(uint32_t val)
    {
        if ( val>max_ || val<min_ )
        {
            if ( val>max_ || val<min_ ) Serial.printf("%s %s set:: out range %ld (%ld, %ld)\n", code_.c_str(), description_.c_str(), val, min_, max_);
            return false;
        }
        else
        {
            *val_ = val;
            if ( is_eeram_ ) rP_->put(addr_.a16, *val_);
            return true;
        }
    }

    virtual void get()
    {
        if ( is_eeram_ )
        {
            rP_->get(addr_.a16, *val_);
        }
    }

    virtual bool is_corrupt()
    {
        bool corrupt = *val_ > max_ || *val_ < min_;
        if ( corrupt ) sendTxBuf(String::format("\n%s %s corrupt", code_.c_str(), description_.c_str()), true, IN_SERVICE); 
        return corrupt;
    }

    virtual bool is_off()
    {
        return off_nominal() && check_for_off_on_init_;
    }

    virtual bool off_nominal()
    {
        return *val_ != default_;
    }

    void print_str()
    {
        sprintf(pr.buff, " %-18s %10d -> %10d, %10s (%s%-2s)", description_.c_str(), (int)default_, (int)*val_, units_.c_str(), prefix_.c_str(), code_.c_str());
    }
    
    void print()
    {
        print_str();
        sendTxBuf(String::format("%s\n", pr.buff), true, IN_SERVICE);
    }
    
    void print_help_str()
    {
        sprintf(pr.buff, "%s%-2s= %6d: (%-6d-%6d) [%6d] %s, %s", prefix_.c_str(), code_.c_str(), (int)*val_, (int)min_, (int)max_, (int)default_, description_.c_str(), units_.c_str());
    }

    void print_help()
    {
        print_help_str();
        sendTxBuf(String::format("%s\n", pr.buff), true, IN_SERVICE);
    }

    virtual bool print_adjust(const String &str)
    {
        print();
        if ( str == "" ) success_ = print_nominalize();
        else success_ = check_set_put((uint32_t) str.toInt());
        print();
        return success_;
    }

    bool print_adj_print(const uint32_t input)
    {
        print();
        success_ = check_set_put(input);
        print();
        return success_;
    }
   
    virtual void set_nominal()
    {
        *val_ = default_;
        if ( is_eeram_ ) rP_->put(addr_.a16, *val_);
    }

protected:
    uint32_t *val_;
    uint32_t min_;
    uint32_t max_;
    uint32_t default_;
};
