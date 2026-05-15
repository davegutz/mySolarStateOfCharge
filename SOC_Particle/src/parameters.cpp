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

#include "application.h"
#include "Battery.h"
#include "parameters.h"
#include "Sensors.h"
#include "command.h"

extern CommandPars cp;


// class Parameters
// Corruption test on bootup.  Needed because retained parameter memory is not managed by the compiler as it relies on
// battery.  Small compilation changes can change where in this memory the program points, too
Parameters::Parameters():n_(0), V_(nullptr), dirty_(false) {};

Parameters::~Parameters(){};

bool Parameters::find_adjust(const String &str)
{
    uint8_t count = 0;
    bool found = false;
    bool success = false;
    String substr = str.substring(0, 2);
    value_str_ = str.substring(2);
    if ( substr.length()<2 )
    {
        Serial.printf("%s substr of %s too short\n", substr.c_str(), str.c_str());
        return false;
    }
    for ( uint8_t i=0; i<n_; i++ )
    {
        if ( substr==V_[i]->code() )
        {
            found = true;
            if ( !count )
            {
                success = V_[i]->print_adjust(value_str_);  // prints own error messages
                if ( success ) dirty_ = true;
            }
            else Serial.printf("RPT: %d %s success=%d\n", i, V_[i]->code().c_str(), success);
            count++;
        }
    }
    if ( found )
    {
        if ( count > 1 )
        {
            Serial.printf("RPT: %s decoded -> code %s and val %s\n", str.c_str(), substr.c_str(), value_str_.c_str());
        }
        return true;
    }
    else
    {
        // Serial.printf("Problem: %s was decoded into code %s and value %s\n", str.c_str(), substr.c_str(), value_str.c_str());
        return false;
    }

}

bool Parameters::is_corrupt()
{
    bool corruption = false;
    for ( int i=0; i<n_; i++ )
    {
        if ( V_[i]->is_corrupt() ) sendTxBuf(String::format("\n%s %s corrupt", V_[i]->code().c_str(), V_[i]->description()), true, IN_SERVICE);
        corruption |= V_[i]->is_corrupt();
    }
    if ( corruption )
    {
        sendTxBuf(String::format("\ncorrupt****\n"), true, IN_SERVICE);
        pretty_print(false);
    }
    return corruption;
}

void Parameters::set_nominal()
{
    for ( uint16_t i=0; i<n_; i++ )  if ( V_[i]->code() != "UT" ) V_[i]->set_nominal();
}


// class VolatilePars
VolatilePars::VolatilePars(): Parameters()
{
    initialize();
    set_nominal();
}

VolatilePars::~VolatilePars(){}

void  VolatilePars::initialize()
{
    #define NVOL 60
    V_ = new Variable*[NVOL];
    V_[n_++] =(cc_diff_slr_p    = new FloatV("  ", "Fc", NULL,"Slr cc_diff thr",      "slr",    0,    1000, &cc_diff_slr_,      1));  // Fc
    V_[n_++] =(cycles_inj_p     = new FloatV("  ", "XC", NULL,"Number prog cycle",    "float",  0,    1000, &cycles_inj_,       0));  // XC
    V_[n_++] =(dc_dc_on_p     = new BooleanV("  ", "Xd", NULL,"DC-DC charger on",     "T=on",   0,    1,    &dc_dc_on_,         false));  // Xd
    V_[n_++] =(disab_ib_fa_p  = new BooleanV("  ", "FI", NULL,"Disab hard range ib",  "T=disab",0,    1,    &disab_ib_fa_,      false));  // FI
    V_[n_++] =(disab_tb_fa_p  = new BooleanV("  ", "FT", NULL,"Disab hard range tb",  "T=disab",0,    1,    &disab_tb_fa_,      DISAB_TB_FA));  // FT
    V_[n_++] =(dis_vb_fa_lt_p  = new BooleanV("  ", "FV", NULL,"Disab hard range vb",  "T=disab",0,    1,   &dis_vb_fa_lt_,     DISAB_VB_FA_LT));  // FV
    V_[n_++] =(ds_voc_soc_p     = new FloatV("  ", "Ds", NULL,"VOC(SOC) del soc",     "slr",    -0.5, 0.5,  &ds_voc_soc_,       NOM_DS));  // Ds
    V_[n_++] =(dv_voc_soc_p     = new FloatV("  ", "Dy", NULL,"VOC(SOC) del v",       "v",      -50,  50,   &dv_voc_soc_,       NOM_DY));  // Dy
    V_[n_++] =(ewhi_slr_p       = new FloatV("  ", "Fi", NULL,"Slr wrap hi thr",      "slr",    0,    1000, &ewhi_slr_,         FI_NOM));  // Fi
    V_[n_++] =(ewlo_slr_p       = new FloatV("  ", "Fo", NULL,"Slr wrap lo thr",      "slr",    0,    1000, &ewlo_slr_,         FO_NOM));  // Fo
    V_[n_++] =(fail_tb_p      = new BooleanV("  ", "Xu", NULL,"Ignore Tb & fail",     "T=Fail", false,true, &fail_tb_,          false));  // Xu
    V_[n_++] =(fake_faults_p  = new BooleanV("  ", "Ff", NULL,"Faults ignored",       "T=ign",  0,    1,    &fake_faults_,      FAKE_FAULTS));  // Ff
    V_[n_++] =(hys_scale_p      = new FloatV("  ", "Sh", NULL,"Sim hys scale",        "slr",    0,    100,  &hys_scale_,        HYS_SCALE));  // Sh
    V_[n_++] =(hys_state_p      = new FloatV("  ", "SH", NULL,"Sim hys state",        "v",      -10,  10,   &hys_state_,        0));  // SH
    V_[n_++] =(Ib_amp_noise_amp_p= new FloatV("  ","DM", NULL,"Amp amp noise",        "A",      0,    1000, &Ib_amp_noise_amp_, IB_AMP_NOISE));  // DM
    V_[n_++] =(ib_amp_add_p     = new FloatV("  ", "Dm", NULL,"Amp signal add",       "A",      -1000,1000, &ib_amp_add_,       0));  // Dm
    V_[n_++] =(ib_max_amp_p     = new FloatV("  ", "Mm", NULL,"Amp hdwe unit max",    "A",      0,    __FLT_MAX__, &ib_amp_max_, (IB_ABS_MAX_AMP/NP/SIZE_MARG)));  // Mm
    V_[n_++] =(ib_min_amp_p     = new FloatV("  ", "Mn", NULL,"Amp hdwe unit min",    "A",      -__FLT_MAX__,   0, &ib_amp_min_, (-IB_ABS_MAX_AMP/NP/SIZE_MARG)));  // Mn
    V_[n_++] =(ib_diff_slr_p    = new FloatV("  ", "Fd", NULL,"Slr ib_diff thr",      "A",      0,    1000, &ib_diff_slr_,      1));  // Fd
    V_[n_++] =(Ib_noa_noise_amp_p= new FloatV("  ","DN", NULL,"Amp noa noise",        "A",      0,    1000, &Ib_noa_noise_amp_, IB_NOA_NOISE));  // DN
    V_[n_++] =(ib_noa_add_p     = new FloatV("  ", "Dn", NULL,"No amp signal add",    "A",      -1000,1000, &ib_noa_add_,       0));  // Dn
    V_[n_++] =(ib_max_noa_p     = new FloatV("  ", "Nm", NULL,"Noa hdwe signal max",  "A",      0,    __FLT_MAX__, &ib_noa_max_, (IB_ABS_MAX_NOA/NP/SIZE_MARG)));  // Nm
    V_[n_++] =(ib_min_noa_p     = new FloatV("  ", "Nn", NULL,"Noa hdwe signal min",  "A",      -__FLT_MAX__,   0, &ib_noa_min_, (-IB_ABS_MAX_NOA/NP/SIZE_MARG)));  // Nn
    V_[n_++] =(ib_quiet_slr_p   = new FloatV("  ", "Fq", NULL,"Ib quiet det slr",     "slr",    0,    1000, &ib_quiet_slr_,     1));  // Fq
    V_[n_++] =(init_all_soc_p   = new FloatV("  ", "Ca", NULL,"Init all to this",     "soc",    -0.5, 1.1,  &init_all_soc_,     1));  // Ca
    V_[n_++] =(init_sim_soc_p   = new FloatV("  ", "Cm", NULL,"Init sim to this",     "soc",    -0.5, 1.1,  &init_sim_soc_,     1));  // Cm
    V_[n_++] =(samp_points_p    = new ULongV("  ", "Cx", NULL,"Number of samples",    "uintl",  0UL,  1000000UL,  &samp_points_, 0));  // Cx
    V_[n_++] =(print_mult_p    = new Uint8tV("  ", "DP", NULL,"Print mult x Dr",      "uint",   1,    UINT8_MAX, &print_mult_,  DP_MULT));  // DP
    V_[n_++] =(read_delay_p     = new ULongV("  ", "Dr", NULL,"Minor frame",          "ms",     1UL,  1000000UL,  &read_delay_, READ_DELAY));  // Dr
    V_[n_++] =(talk_delay_p     = new ULongV("  ", "D>", NULL,"Talk frame",           "ms",     1UL,  120000UL,&talk_delay_,   TALK_DELAY));  // D>
    V_[n_++] =(sum_delay_p      = new ULongV("  ", "Dh", NULL,"Summary frame",        "ms",    1000UL,SUMMARY_DELAY,&sum_delay_, SUMMARY_DELAY));  // Dh
    V_[n_++] =(eframe_mult_p   = new Uint8tV("  ", "ED", NULL,"EKF frame rate x Dr",  "uint",   1,    UINT8_MAX, &eframe_mult_, EKF_EFRAME_MULT));  // ED
    V_[n_++] =(slr_res_p        = new FloatV("  ", "Sr", NULL,"Scalar Randles R0",    "slr",    0,    100,  &slr_res_,         1));  // Sr
    V_[n_++] =(s_t_sat_p        = new FloatV("  ", "Xs", NULL,"Scalar on T_SAT",      "slr",    0,    100,  &s_t_sat_,         1));  // Xs
    V_[n_++] =(tail_inj_p       = new ULongV("  ", "XT", NULL,"Tail end inj",         "ms",     0UL,  120000UL,&tail_inj_,     0UL));  // XT
    V_[n_++] =(Tb_bias_model_p  = new FloatV("  ", "D^", NULL,"Del model",            "dg C",   -120, 50,   &Tb_bias_model_,   TEMP_BIAS));  // D^
    V_[n_++] =(Tb_noise_amp_p   = new FloatV("  ", "DT", NULL,"Tb noise",             "dg C pk-pk", 0,50,   &Tb_noise_amp_,    TB_NOISE));  // DT
    V_[n_++] =(tb_stale_time_slr_p=new FloatV("  ","Xv", NULL,"Scale Tb 1-wire pers", "slr",    0,    100,  &tb_stale_time_slr_,1));  // Xv
    V_[n_++] =(until_q_p        = new ULongV("  ", "XQ", NULL,"Time until vv0",       "ms",     0UL,  1000000UL,  &until_q_,   0UL));  // XQ
    V_[n_++] =(vb_add_p         = new FloatV("  ", "Dv", NULL,"Bias on vb",           "v",      -15,  15,   &vb_add_,          0));  // Dv
    V_[n_++] =(Vb_noise_amp_p   = new FloatV("  ", "DV", NULL,"Vb noise",             "v pk-pk",0,    10,   &Vb_noise_amp_,    VB_NOISE));  // DV
    V_[n_++] =(vc_add_p         = new FloatV("  ", "D3", NULL,"Bias on Vc/Vr",        "v",     -1.65, 0.85, &vc_add_,          0));  // D3
    V_[n_++] =(wait_inj_p       = new ULongV("  ", "XW", NULL,"Wait start inj",       "ms",     0UL,  120000UL, &wait_inj_,    0UL));  // XW
    V_[n_++] =(voc_stat_filt_p  = new FloatV("  ", "VF", NULL,"voc_stat_f time",      "s",      1,    180,  &voc_stat_filt_,   VOC_STAT_FILT));  // VF
    V_[n_++] =(Tb_filt_p        = new FloatV("  ", "VT", NULL,"Tb_f time",            "s",      1,    180,  &Tb_filt_,         TB_FILT));  // VT
    V_[n_++] =(ekf_q_p          = new FloatV("  ", "VQ", NULL,"EKF_Q_SD_NORM volt",   "slr",    0,    10000,&ekf_q_,           1));  // VQ
    V_[n_++] =(ekf_r_p          = new FloatV("  ", "VR", NULL,"EKF_R_SD_NORM frac",   "slr",    0,    10000,&ekf_r_,           1));  // VR
    V_[n_++] =(ekf_conv_p       = new FloatV("  ", "VC", NULL,"ekf conv abs",         "v",      0,    1,    &ekf_conv_,        EKF_CONV));  // V:C
    V_[n_++] =(ekf_x_p          = new FloatV("  ", "Ce", NULL,"ekf x manual set",     "soc",    0,    1,    &ekf_x_,           0));  // Ce
    V_[n_++] =(ekf_p_p          = new FloatV("  ", "Cp", NULL,"ekf P manual set",     "?",     -1e12, 1e12, &ekf_p_,           0));
    V_[n_++] =(q_std_p          = new FloatV("  ", "Kq", NULL,"kf q_std process",     "v",      0,    1e12, &q_std_,           KF_Q_STD));  // Kq
    V_[n_++] =(r_std_p          = new FloatV("  ", "Kr", NULL,"kf_r_std state",       "v",      0,    1e12, &r_std_,           KF_R_STD));  // Kr
    V_[n_++] =(ib_scale_amp_p   = new FloatV("  ", "SA", NULL,"Slr amp",              "A",      -1e5, 1e5,  &ib_scale_amp_,    CURR_SCALE_AMP));  // SA
    V_[n_++] =(ib_scale_noa_p   = new FloatV("  ", "SB", NULL,"Slr noa",              "A",      -1e5, 1e5,  &ib_scale_noa_,    CURR_SCALE_NOA));  // SB
    V_[n_++] =(nP_p             = new FloatV("  ", "BP", NULL,"Number parallel",      "units",  1e-6, 100,  &nP_,          NP));  // BP
    V_[n_++] =(nS_p             = new FloatV("  ", "BS", NULL,"Number series",        "units",  1e-6, 100,  &nS_,          NS));  // BS
    V_[n_++] =(s_cap_mon_p      = new FloatV("  ", "SQ", NULL,"Scalar cap Mon",       "slr",    0,    1000, &s_cap_mon_,   1.));  // SQ
    V_[n_++] =(s_cap_sim_p      = new FloatV("  ", "Sq", NULL,"Scalar cap Sim",       "slr",    0,    1000, &s_cap_sim_,   1.));  // Sq
    V_[n_++] =(Vb_scale_p       = new FloatV("  ", "SV", NULL,"Scale Vb sensor",      "v",      -1e5, 1e5,  &Vb_scale_,    VB_SCALE));  // SV
    V_[n_++] =(snap_wait_p      = new ULongV("  ", "SW", NULL,"Snap wait",            "ms",     0UL,  10000UL,  &snap_wait_,   SNAP_WAIT));  // SW
}


void VolatilePars::pretty_print(const bool all)
{
    #ifndef SOFT_DEPLOY_PHOTON
        if ( all )
        {
            sendTxBuf("volatile all:\n", true, IN_SERVICE);
            for (uint8_t i=0; i<n_; i++ )
            {
                if ( !(V_[i]->is_eeram()) )
                {
                    V_[i]->print();
                }
            }
        }
    #endif
    if ( !all )
    {
        sendTxBuf("volatile off:\n", true, IN_SERVICE);
        uint8_t count = 0;
        for (uint8_t i=0; i<n_; i++ )
        {
            if ( !(V_[i]->is_eeram()) )
            {
                if ( all || V_[i]->is_off() )
                {
                    count++;
                    V_[i]->print();
                }
            }
        }
        if ( count==0 ) sendTxBuf("**none**\n\n", true, IN_SERVICE);
    }
    while ( n_ != NVOL ) { delay(5000); sendTxBuf(String::format("set NVOL=%d\n", n_), true, IN_SERVICE); }
}


/* Using pointers in building class so all that stuff does not get saved by 'retained' keyword in SOC_Particle.ino.
    Only the *_z parameters at the bottom of Parameters.h are stored in SRAM
*/
// class SavedPars 
SavedPars::SavedPars(): Parameters()
{
    nflt_ = uint16_t( NFLT ); 
    nhis_ = uint16_t( NHIS );
    nsum_ = 0;
}

SavedPars::SavedPars(Flt_st *hist, const uint16_t nhis, Flt_st *faults, const uint16_t nflt): Parameters()
{
    rP_ = NULL;
    nflt_ = nflt;
    nhis_ = nhis;
    nsum_ = 0;
    history_ = hist;
    fault_ = faults;
    initialize();
}

SavedPars::SavedPars(SerialRAM *ram): Parameters()
{
    rP_ = ram;
    next_ = 0x000;
    nflt_ = uint16_t( NFLT ); 
    initialize();

    // Don't nominalize SavedPars on load.  Defeats the whole purpose of EERAM
    // for ( uint8_t i=0; i<n_; i++ ) if ( !V_[i]->is_eeram() ) V_[i]->set_nominal();  no!!

}

SavedPars::~SavedPars() {}

void SavedPars::initialize()
{
    #define NSAV 24
    V_ = new Variable*[NSAV];
    V_[n_++] =(amp_p            = new FloatV("* ", "Xa", rP_, "Inj amp",              "Amps pk",-1e6, 1e6,  &amp_,              0));  // Xa
    V_[n_++] =(booted_p       = new BooleanV("  ", "Bb", rP_, "Clean boot",       "T=clean",     0,    1,   &booted_,           false));  // Bb
    V_[n_++] =(cutback_gain_slr_p=new FloatV("* ", "Sk", rP_, "Cutback gain scalar",  "slr",    -1e6, 1e6,  &cutback_gain_slr_, 1));  // Sk
    V_[n_++] =(debug_p            = new IntV("* ", "vv", rP_, "Verbosity",            "int",    -128, 128,  &debug_,            VV));  // vv
    V_[n_++] =(delta_q_model_p = new DoubleV("* ", "qs", rP_, "Charge chg Sim",       "C",      -1e8, 1e5,  &delta_q_model_,    0, false));   // qs
    V_[n_++] =(delta_q_p       = new DoubleV("* ", "qm", rP_, "Charge chg",           "C",      -1e8, 1e5,  &delta_q_,          0, false ));  // qm
    V_[n_++] =(Dw_p             = new FloatV("* ", "Dw", rP_, "Tab mon adj",          "v",      -1e2, 1e2,  &Dw_,               VTAB_BIAS));  // Dw
    V_[n_++] =(freq_p           = new FloatV("* ", "Xf", rP_, "Inj freq",             "Hz",      0,   2,    &freq_,             0));  // Xf
    V_[n_++] =(ib_bias_all_p    = new FloatV("* ", "DI", rP_, "Del all",              "A",      -1e5, 1e5,  &ib_bias_all_,      CURR_BIAS_ALL));  // DI
    V_[n_++] =(ib_bias_amp_p    = new FloatV("* ", "DA", rP_, "Add amp",              "A",      -1e5, 1e5,  &ib_bias_amp_,      CURR_BIAS_AMP));  // DA
    V_[n_++] =(ib_bias_noa_p    = new FloatV("* ", "DB", rP_, "Add noa",              "A",      -1e5, 1e5,  &ib_bias_noa_,      CURR_BIAS_NOA));  // DB
    V_[n_++] =(ib_disch_slr_p   = new FloatV("* ", "SD", rP_, "Slr disch",            "slr",    -1e5, 1e5,  &ib_disch_slr_,     CURR_SCALE_DISCH));  // SD
    #ifdef HDWE_IB_HI_LO
        V_[n_++] =(ib_force_p      = new Int8tV("* ", "si", rP_, "curr sel mode",        "(-1, 0, 1)", -1, 1,  &ib_force_,      int8_t(IB_FORCE)));  // si
    #else
        V_[n_++] =(ib_force_p      = new Int8tV("* ", "si", rP_, "curr sel mode",        "(-1, 0, 1)", -1, 1,  &ib_force_,      int8_t(FAKE_FAULTS)));  // si
    #endif
    V_[n_++] =(iflt_p         = new Uint16tV("* ", "if", rP_, "Fault buffer indx",    "uint",   0,nflt_+1,  &iflt_,             nflt_,      false));  // if
    V_[n_++] =(ihis_p         = new Uint16tV("* ", "ih", rP_, "Hist buffer indx",     "uint",   0,nhis_+1,  &ihis_,             nhis_,      false));  // ih
    V_[n_++] =(inj_bias_p       = new FloatV("* ", "Xb", rP_, "Injection bias",       "A",      -1e5, 1e5,  &inj_bias_,         0.));               // Xb
    V_[n_++] =(isum_p         = new Uint16tV("* ", "is", rP_, "Summ buffer indx",     "uint",   0, NSUM+1,  &isum_,             NSUM,       false));  // is
    V_[n_++] =(modeling_p      = new Uint8tV("* ", "Xm", rP_, "Modeling bitmap",      "[0x]",   0,    255,  &modeling_,         MODELING));         // Xm
    V_[n_++] =(preserving_p    = new Uint8tV("* ", "X?", rP_, "Preserving fault",     "T=Preserve",0,   1,  &preserving_,       0,          false));  // X?
    V_[n_++] =(Tb_bias_hdwe_p   = new FloatV("* ", "Dt", rP_, "Bias Tb sensor",       "dg C",   -500, 500,  &Tb_bias_hdwe_,     TEMP_BIAS));        // Dt
    V_[n_++] =(Time_now_p       = new ULongV("* ", "UT", rP_, "UNIX time epoch",      "sec",    0UL,  2100000000UL, &Time_now_, 1669801880UL,  false));  // UT
    V_[n_++] =(Type_p          = new Uint8tV("* ", "Xt", rP_, "Inj type",             "1sn 2sq 3tr 4 1C, 5 -1C, 8cs",  0,   10,  &type_,    0));  // Xt
    V_[n_++] =(Vb_bias_hdwe_p   = new FloatV("* ", "Dc", rP_, "Bias Vb sensor",       "v",      -10,  70,   &Vb_bias_hdwe_,     VOLT_BIAS));  // Dc
    V_[n_++] =(vsat_add_p       = new FloatV("* ", "DS", rP_, "Bias on nominal vsat", "v",      -2.,  2.,   &vsat_add_,         VSAT_ADD));  // DS
}

// Number of differences between nominal EERAM and actual (don't count integator memories because they always change)
int SavedPars::num_diffs()
{
    int n = 0;
    for (int i=0; i<n_; i++ ) if ( V_[i]->is_off() )  n++;
    return ( n );
}

// Configuration functions

// Print memory map
void SavedPars::mem_print()
{
}

// Print
void SavedPars::pretty_print(const bool all)
{
    if ( all )
    {
        sendTxBuf("saved (sp) all\n", true, IN_SERVICE);
        for (int i=0; i<n_; i++ )
        {
            V_[i]->print();
        }
        #ifndef SOFT_DEPLOY_PHOTON
            sendTxBuf("Xm:\n", true, IN_SERVICE);
            pretty_print_modeling();
        #endif
    }
    else
    {
        Serial.printf("saved (sp) diffs\n");
        uint8_t count = 0;
        for (int i=0; i<n_; i++ )
        {
            if ( V_[i]->is_off() )
            {
                count++;
                V_[i]->print();
            }
        }
        if ( count==0 ) sendTxBuf("**none**\n\n", true, IN_SERVICE);

        // Build integrity test
        while ( n_ != NSAV ) { delay(5000); sendTxBuf(String::format("set NSAV=%d\n", n_), true, IN_SERVICE); }
    }
}

void SavedPars::pretty_print_modeling()
{
  char buffer[32];
  bitMapPrint(buffer, sp.modeling(), 8);
  Serial.printf(" 0x%s\n", buffer);
  Serial.printf(" 0x128 ib_noa_dscn %d\n", mod_ib_noa_dscn());
  Serial.printf(" 0x64  ib_amp_dscn %d\n", mod_ib_amp_dscn());
  Serial.printf(" 0x32  vb_dscn %d\n", mod_vb_dscn());
  Serial.printf(" 0x16  temp_dscn %d\n", mod_tb_dscn());
  Serial.printf(" 0x8   tweak_test %d\n", tweak_test());
  Serial.printf(" 0x4   current %d\n", mod_ib());
  Serial.printf(" 0x2   voltage %d\n", mod_vb());
  Serial.printf(" 0x1   temp %d\n", mod_tb());

  
  time_long_2_str((time_t)Time_now_, buffer);
  sendTxBuf(String::format(" time %ld hms:  %s\n", Time_now_, buffer), true, IN_SERVICE);
}

// Print faults
void SavedPars::print_fault_array()
{
  uint16_t i = iflt_;  // Last one written was iflt
  uint16_t n = 0;
  while ( ++n < nflt_+1 )
  {
    if ( ++i > (nflt_-1) ) i = 0; // circular buffer
    fault_[i].print_flt("unit_f");
  }
}

// Print history
void SavedPars::print_history_array()
{
  int i = ihis_;  // Last one written was ihis_
  int n = -1;
  while ( ++n < nhis_ )
  {
    if ( ++i > (nhis_-1) ) i = 0; // circular buffer
    history_[i].print_flt("unit_h");
  }
}

// Dynamic parameters saved
// This saves a lot of througput.   Without it, there are many put calls each 'read' minor frame at 1 ms each call
void SavedPars::put_all_dynamic()
{
    static uint8_t blink = 0;
    switch ( blink++ )
    {
        case ( 0 ):
            put_delta_q();
            break;

        case ( 1 ):
            put_delta_q_model();
            break;

        case ( 2 ):
            put_Time_now(max( Time_now_, (uint32_t)Time.now()));  // If happen to connect to wifi (assume updated automatically), save new time
            blink = 0;
            break;

        default:
            blink = 0;
            break;
    }
}
 
 // Bounce history elements
Flt_st SavedPars::put_history(Flt_st input, const uint8_t i)
{
    Flt_st bounced_sum;
    bounced_sum.copy_to_Flt_ram_from(history_[i]);
    history_[i].put(input);
    return bounced_sum;
}

// Reset arrays
void SavedPars::reset_flt()
{
    for ( uint16_t i=0; i<nflt_; i++ )
    {
        fault_[i].put_nominal();
    }
 }
void SavedPars::reset_his()
{
    for ( uint16_t i=0; i<nhis_; i++ )
    {
        history_[i].put_nominal();
    }
 }

void SavedPars::set_nominal()
{
    Parameters::set_nominal();

    put_Inj_bias(float(0.));

    put_Preserving(uint8_t(0));
 }

