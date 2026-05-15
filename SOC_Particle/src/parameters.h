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

#include "constants.h"
#include "Battery.h"
#include "FaultStore.h"
#include "PrinterPars.h"
#include "Variable.h"
#include "Cloud.h"


class Parameters
{
public:
    Parameters();
    ~Parameters();
    // Do everything
    bool dirty() { return dirty_; }
    void dirty(const bool d) { dirty_ = d; }
    bool find_adjust(const String &str);
    virtual void initialize() {}
    bool is_corrupt();
    virtual void pretty_print(const bool all){}
    void set_nominal();
    String value_str() { return value_str_; }
protected:
    int8_t n_;
    Variable **V_;
    String value_str_;
    bool dirty_;  // set when a retained param is modified; cleared on backupRamSync (talk 'w')
};


// Volatile memory
class VolatilePars : public Parameters
{
public:
    VolatilePars();
    ~VolatilePars();
    virtual void initialize();
    virtual void pretty_print(const bool all);

    FloatV *cc_diff_slr_p;
    FloatV *cycles_inj_p;
    BooleanV *dc_dc_on_p;
    BooleanV *disab_ib_fa_p;
    BooleanV *disab_tb_fa_p;
    BooleanV *dis_vb_fa_lt_p;
    FloatV *ds_voc_soc_p;
    FloatV *dv_voc_soc_p;
    Uint8tV *eframe_mult_p;
    FloatV *ewhi_slr_p;
    FloatV *ewlo_slr_p;
    BooleanV *fail_tb_p;
    BooleanV *fake_faults_p;
    FloatV *hys_scale_p;
    FloatV *hys_state_p;
    FloatV *ib_amp_add_p;
    FloatV *ib_diff_slr_p;
    FloatV *ib_noa_add_p;
    FloatV *Ib_amp_noise_amp_p;
    FloatV *Ib_noa_noise_amp_p;
    FloatV *ib_quiet_slr_p;
    FloatV *init_all_soc_p;
    FloatV *init_sim_soc_p;
    Uint8tV *print_mult_p;
    ULongV *read_delay_p;
    ULongV *samp_points_p;
    FloatV *slr_res_p;
    FloatV *s_t_sat_p;
    ULongV *sum_delay_p;
    ULongV *tail_inj_p;
    ULongV *talk_delay_p;
    FloatV *Tb_bias_model_p;
    FloatV *Tb_noise_amp_p;
    FloatV *tb_stale_time_slr_p;
    ULongV *until_q_p;
    FloatV *vb_add_p;
    FloatV *Vb_noise_amp_p;
    FloatV *vc_add_p;
    ULongV *wait_inj_p;
    FloatV *ib_max_amp_p;
    FloatV *ib_min_amp_p;
    FloatV *ib_max_noa_p;
    FloatV *ib_min_noa_p;
    FloatV *voc_stat_filt_p;
    FloatV *Tb_filt_p;
    FloatV *ekf_q_p;
    FloatV *ekf_r_p;
    FloatV *ekf_conv_p;
    FloatV *ekf_x_p;
    FloatV *ekf_p_p;
    FloatV *q_std_p;
    FloatV *r_std_p;
    FloatV *ib_scale_amp_p; 
    FloatV *ib_scale_noa_p;
    FloatV *nP_p;
    FloatV *nS_p;
    FloatV *s_cap_mon_p;
    FloatV *s_cap_sim_p;
    ULongV *snap_wait_p;
    FloatV *Vb_scale_p;

    // accessors
    float cc_diff_slr() { return cc_diff_slr_; }
    void cc_diff_slr(const float input) { cc_diff_slr_ = input; }
    float cycles_inj() { return cycles_inj_; }
    void cycles_inj(const float input) { cycles_inj_ = input; }
    bool dc_dc_on() { return dc_dc_on_; }
    void dc_dc_on(const bool input) { dc_dc_on_ = input; }
    bool disab_ib_fa() { return disab_ib_fa_; }
    void disab_ib_fa(const bool input) { disab_ib_fa_ = input; }
    bool disab_tb_fa() { return disab_tb_fa_; }
    void disab_tb_fa(const bool input) { disab_tb_fa_ = input; }
    bool disab_vb_fa_lt() { return dis_vb_fa_lt_; }
    void disab_vb_fa_lt(const bool input) { dis_vb_fa_lt_ = input; }
    float ds_voc_soc() { return ds_voc_soc_; }
    void ds_voc_soc(const float input) { ds_voc_soc_ = input; }
    float dv_voc_soc() { return dv_voc_soc_; }
    void dv_voc_soc(const float input) { dv_voc_soc_ = input; }
    uint8_t eframe_mult() { return eframe_mult_; }
    void eframe_mult(const uint8_t input) { eframe_mult_ = input; }
    float ekf_conv() { return ekf_conv_; }
    void ekf_conv(const float input) { ekf_conv_ = input; }
    float ekf_p() { return ekf_p_; }
    void ekf_p(const float input) { ekf_p_ = input; }
    float ekf_q() { return ekf_q_; }
    void ekf_q(const float input) { ekf_q_ = input; }
    float ekf_r() { return ekf_r_; }
    void ekf_r(const float input) { ekf_r_ = input; }
    float ekf_x() { return ekf_x_; }
    void ekf_x(const float input) { ekf_x_ = input; }
    float ewhi_slr() { return ewhi_slr_; }
    void ewhi_slr(const float input) { ewhi_slr_ = input; }
    float ewlo_slr() { return ewlo_slr_; }
    void ewlo_slr(const float input) { ewlo_slr_ = input; }
    bool fail_tb() { return fail_tb_; }
    void fail_tb(const bool input) { fail_tb_ = input; }
    bool fake_faults() { return fake_faults_; }
    void fake_faults(const bool input) { fake_faults_ = input; }
    float hys_scale() { return hys_scale_; }
    void hys_scale(const float input) { hys_scale_ = input; }
    float hys_state() { return hys_state_; }
    void hys_state(const float input) { hys_state_ = input; }
    float ib_amp_add() { return ib_amp_add_; }
    void ib_amp_add(const float input) { ib_amp_add_ = input; }
    float ib_amp_max() { return ib_amp_max_; }
    void ib_amp_max(const float input) { ib_amp_max_ = input; }
    float ib_amp_min() { return ib_amp_min_; }
    void ib_amp_min(const float input) { ib_amp_min_ = input; }
    float Ib_amp_noise_amp() { return Ib_amp_noise_amp_; }
    void Ib_amp_noise_amp(const float input) { Ib_amp_noise_amp_ = input; }
    float ib_diff_slr() { return ib_diff_slr_; }
    void ib_diff_slr(const float input) { ib_diff_slr_ = input; }
    float ib_noa_add() { return ib_noa_add_; }
    void ib_noa_add(const float input) { ib_noa_add_ = input; }
    float ib_noa_max() { return ib_noa_max_; }
    void ib_noa_max(const float input) { ib_noa_max_ = input; }
    float ib_noa_min() { return ib_noa_min_; }
    void ib_noa_min(const float input) { ib_noa_min_ = input; }
    float Ib_noa_noise_amp() { return Ib_noa_noise_amp_; }
    void Ib_noa_noise_amp(const float input) { Ib_noa_noise_amp_ = input; }
    float ib_quiet_slr() { return ib_quiet_slr_; }
    void ib_quiet_slr(const float input) { ib_quiet_slr_ = input; }
    float ib_scale_amp() { return ib_scale_amp_; }
    void ib_scale_amp(const float input) { ib_scale_amp_ = input; }
    float *ib_scale_amp_ptr() { return &ib_scale_amp_; }
    float ib_scale_noa() { return ib_scale_noa_; }
    void ib_scale_noa(const float input) { ib_scale_noa_ = input; }
    float *ib_scale_noa_ptr() { return &ib_scale_noa_; }
    float init_all_soc() { return init_all_soc_; }
    void init_all_soc(const float input) { init_all_soc_ = input; }
    float init_sim_soc() { return init_sim_soc_; }
    void init_sim_soc(const float input) { init_sim_soc_ = input; }
    float nP() { return nP_; }
    void nP(const float input) { nP_ = input; }
    float nS() { return nS_; }
    void nS(const float input) { nS_ = input; }
    uint8_t print_mult() { return print_mult_; }
    void print_mult(const uint8_t input) { print_mult_ = input; }
    float q_std() { return q_std_; }
    void q_std(const float input) { q_std_ = input; }
    float r_std() { return r_std_; }
    void r_std(const float input) { r_std_ = input; }
    uint32_t read_delay() { return read_delay_; }
    void read_delay(const uint32_t input) { read_delay_ = input; }
    float s_cap_mon() { return s_cap_mon_; }
    void s_cap_mon(const float input) { s_cap_mon_ = input; }
    float s_cap_sim() { return s_cap_sim_; }
    void s_cap_sim(const float input) { s_cap_sim_ = input; }
    float s_t_sat() { return s_t_sat_; }
    void s_t_sat(const float input) { s_t_sat_ = input; }
    uint32_t samp_points() { return samp_points_; }
    void samp_points(const uint32_t input) { samp_points_ = input; }
    float slr_res() { return slr_res_; }
    void slr_res(const float input) { slr_res_ = input; }
    uint32_t sum_delay() { return sum_delay_; }
    void sum_delay(const uint32_t input) { sum_delay_ = input; }
    uint32_t tail_inj() { return tail_inj_; }
    void tail_inj(const uint32_t input) { tail_inj_ = input; }
    uint32_t talk_delay() { return talk_delay_; }
    void talk_delay(const uint32_t input) { talk_delay_ = input; }
    float Tb_bias_model() { return Tb_bias_model_; }
    void Tb_bias_model(const float input) { Tb_bias_model_ = input; }
    float Tb_filt() { return Tb_filt_; }
    void Tb_filt(const float input) { Tb_filt_ = input; }
    float Tb_noise_amp() { return Tb_noise_amp_; }
    void Tb_noise_amp(const float input) { Tb_noise_amp_ = input; }
    float tb_stale_time_slr() { return tb_stale_time_slr_; }
    void tb_stale_time_slr(const float input) { tb_stale_time_slr_ = input; }
    uint32_t until_q() { return until_q_; }
    void until_q(const uint32_t input) { until_q_ = input; }
    float vb_add() { return vb_add_; }
    void vb_add(const float input) { vb_add_ = input; }
    float Vb_noise_amp() { return Vb_noise_amp_; }
    void Vb_noise_amp(const float input) { Vb_noise_amp_ = input; }
    float Vb_scale() { return Vb_scale_; }
    void Vb_scale(const float input) { Vb_scale_ = input; }
    float vc_add() { return vc_add_; }
    void vc_add(const float input) { vc_add_ = input; }
    float voc_stat_filt() { return voc_stat_filt_; }
    void voc_stat_filt(const float input) { voc_stat_filt_ = input; }
    uint32_t wait_inj() { return wait_inj_; }
    void wait_inj(const uint32_t input) { wait_inj_ = input; }

    // Put into RAM and check for validity
    void put_ib_scale_amp(const float input) { ib_scale_amp_p->check_set_put(input); }
    void put_ib_scale_noa(const float input) { ib_scale_noa_p->check_set_put(input); }
    void put_nP(const float input) { nP_p->check_set_put(input); }
    void put_nS(const float input) { nS_p->check_set_put(input); }
    void put_s_cap_mon(const float input) { s_cap_mon_p->check_set_put(input); }
    void put_s_cap_sim(const float input) { s_cap_sim_p->check_set_put(input); }
    void put_Vb_scale(const float input) { Vb_scale_p->check_set_put(input); }

protected:

    float cc_diff_slr_;          // Scale cc_diff detection thresh, scalar
    float cycles_inj_;           // Number of injection cycles
    bool dc_dc_on_;           // DC-DC charger is on
    bool dis_vb_fa_lt_;        // Disable hard fault range failures for vb
    bool disab_ib_fa_;        // Disable hard fault range failures for ib
    bool disab_tb_fa_;        // Disable hard fault range failures for tb
    float ds_voc_soc_;           // VOC(SOC) delta soc on input, frac
    float dv_voc_soc_;           // VOC(SOC) del v, V
    uint8_t eframe_mult_;        // Frame multiplier for EKF execution.  Number of READ executes for each EKF execution
    float ekf_conv_;             // ekf abs conv, v
    float ekf_p_;                // ekf temporary set P, soc
    float ekf_q_;                // ekf_q scalar, slr
    float ekf_r_;                // ekf_r scalar, slr
    float ekf_x_;                // ekf temporary set x, soc
    float ewhi_slr_;             // Scale wrap hi detection thresh, scalar
    float ewlo_slr_;             // Scale wrap lo detection thresh, scalar
    bool fail_tb_;            // Make hardware bus read ignore Tb and fail it
    bool fake_faults_;        // Faults faked (ignored).  Used to evaluate a configuration, deploy it without disrupting use
    float hys_scale_;            // Sim hysteresis scalar
    float hys_state_;            // Sim hysteresis state
    float ib_amp_add_;           // Fault injection bias on amp, A
    float ib_amp_max_;           // ib amp unit hardware model max, A
    float ib_amp_min_;           // ib amp unit hardware model min, A
    float Ib_amp_noise_amp_;     // Ib bank noise on amplified sensor, amplitude model only, A pk-pk
    float ib_diff_slr_;          // Scale ib_diff detection thresh, scalar
    float ib_noa_add_;           // Fault injection bias on non amp, A
    float ib_noa_max_;           // ib noa unit hardware model max, A
    float ib_noa_min_;           // ib noa unit hardware model min, A
    float Ib_noa_noise_amp_;     // Ib bank noise on non-amplified sensor, amplitude model only, A pk-pk
    float ib_quiet_slr_;         // Scale ib_quiet detection thresh, scalar
    float ib_scale_amp_;         // Slr amp
    float ib_scale_noa_;         // Slr noa
    float init_all_soc_;         // Reinitialize all models to this soc
    float init_sim_soc_;         // Reinitialize sim model only to this soc
    float nP_;                   // number parallel
    float nS_;                   // number series
    uint8_t print_mult_;         // Print multiplier for objects
    float q_std_;                // kf q_std set, v
    float r_std_;                // kf q_std set, v
    uint32_t read_delay_; // Minor frame, ms
    float s_cap_mon_;            // Scalar cap Mon
    float s_cap_sim_;            // Scalar cap Sim
    float s_t_sat_;              // Scalar on saturation test time set and reset
    uint32_t samp_points_; // Number of sample readings to take, !=0 initiates sampling
    float slr_res_;              // Scalar Randles R0, slr
    uint32_t sum_delay_; // Minor frame divisor, div
    uint32_t snap_wait_; // Wait after snap before print, ms
    uint32_t tail_inj_; // Tail after end injection, ms
    uint32_t talk_delay_; // Talk frame, ms
    float Tb_bias_model_;        // Bias on Tb for model
    float Tb_filt_;              // TbHdweFilt time constant, s
    float Tb_noise_amp_;         // Tb noise amplitude model only, deg C pk-pk
    float tb_stale_time_slr_;    // Scalar on persistences of Tb hardware stale check
    uint32_t until_q_;  // Time until set vv0, ms
    float vb_add_;               // Fault injection bias, V
    float Vb_noise_amp_;         // Vb bank noise amplitude model only, V pk-pk
    float Vb_scale_;             // Scale Vb sensor
    float vc_add_;               // Shunt Vc/Vr Fault injection bias, V
    float voc_stat_filt_;        // VocStatFilt time constant, s
    uint32_t wait_inj_; // Wait before start injection, ms

};


// Definition of structure to be saved, either EERAM or retained backup SRAM.  Many are needed to calibrate.  Others are
// needed to allow testing with resets.  Others allow application to remember dynamic
// tweaks.  Default values below are important:  they prevent junk
// behavior on initial build. Don't put anything in here that you can't live with normal running
// because could get set by testing and forgotten.
// SavedPars Class
class SavedPars : public Parameters
{
public:
    SavedPars();
    SavedPars(SerialRAM *ram);
    SavedPars(Flt_st *hist, const uint16_t nhis, Flt_st *faults, const uint16_t nflt);
    ~SavedPars();
 
    // parameter list
    float Amp(const float nP) { return amp_ * nP; }
    bool booted() { return booted_; }
    void booted(const bool input) { booted_ = input; }
    float cutback_gain_slr() { return cutback_gain_slr_; }
    int debug() { return debug_;}
    double delta_q() { return delta_q_;}
    double *delta_q_ptr() { return &delta_q_;}
    double delta_q_model() { return delta_q_model_;}
    double *delta_q_model_ptr() { return &delta_q_model_;}
    float Dw() { return Dw_; }
    float freq() { return freq_; }
    void freq(const float input) { freq_ = input; }
    uint16_t iflt() { return iflt_; }
    uint16_t ihis() { return ihis_; }
    float ib_bias_all() { return ib_bias_all_; }
    void ib_bias_all(const float input) { ib_bias_all_ = input; }
    float ib_bias_amp() { return ib_bias_amp_; }        // DA
    float *ib_bias_amp_ptr() { return &ib_bias_amp_; }  // DA
    float ib_bias_noa() { return ib_bias_noa_; }        // DB
    float *ib_bias_noa_ptr() { return &ib_bias_noa_; }  // DB
    float ib_disch_slr() { return ib_disch_slr_; }
    int8_t ib_force() { return ib_force_; }
    float inj_bias() { return inj_bias_; }
    uint16_t isum() { return isum_; }
    uint8_t modeling() { return modeling_; }
    uint8_t *modeling_ptr() { return &modeling_; }
    uint8_t preserving() { return preserving_; }
    uint8_t *preserving_ptr() { return &preserving_; }
    float Tb_bias_hdwe() { return Tb_bias_hdwe_; }
    uint32_t Time_now() { return Time_now_; }
    uint8_t type() { return type_; }
    float Vb_bias_hdwe() { return Vb_bias_hdwe_; }
    float Vsat_add() { return vsat_add_; }

    // functions
    virtual void initialize();
    void large_reset() { set_nominal(); reset_flt(); reset_his(); }
    void mem_print();
    uint16_t nflt() { return nflt_; }
    uint16_t nhis() { return nhis_; }
    void nsum(const uint16_t in) { nsum_ = in; }
    uint16_t nsum() { return nsum_; }
    void nominalize_fault_array();
    void nominalize_history_array();
    int num_diffs();
    virtual void pretty_print(const bool all);
    void pretty_print_modeling();
    void print_fault_array();
    void print_fault_header(Publish *pubList);
    void print_history_array();
    void reset_flt();
    void reset_his();
    virtual void set_nominal();
    float ib_hist_m_slr() { if ( abs(amp_) > SCL_40 ) return SCL_30000/abs(amp_); else return SCL_600; }
    float ib_hist_n_slr() { if ( abs(amp_) > SCL_40 ) return SCL_30000/abs(amp_); else return SCL_60; }
    float vb_hist_slr() { if ( abs(amp_) > SCL_40 ) return SCL_1500/abs(amp_); else return SCL_1200; }
    bool mod_all_dscn() { return ( 111<modeling() ); }                // Bare all
    bool mod_any() { return ( mod_ib() || mod_tb() || mod_vb() ); }  // Modeling any
    bool mod_none_dscn() { return ( 16>modeling() ); }                // Bare nothing
    bool mod_any_dscn() { return ( 15<modeling() ); }                 // Bare any
    bool mod_ib_all_dscn() { return ( 191<modeling() ); }             // Nothing connected to ib sensors in I2C on SDA/SCL
    bool mod_ib_any_dscn() { return ( mod_ib_amp_dscn() || mod_ib_noa_dscn() ); }  // Nothing connected to ib sensors in I2C on SDA/SCL
    bool mod_ib_noa_dscn() { return ( 1<<7 & modeling() ); }          // Nothing connected to noa ib sensors in I2C on SDA/SCL
    bool mod_ib_amp_dscn() { return ( 1<<6 & modeling() ); }          // Nothing connected to amp ib sensors in I2C on SDA/SCL
    bool mod_vb_dscn() { return ( 1<<5 & modeling() ); }              // Nothing connected to vb on A1
    bool mod_tb_dscn() { return ( 1<<4 & modeling() ); }              // Nothing connected to one-wire Tb sensor on D6
    bool mod_ib() { return ( 1<<2 & modeling() || mod_ib_all_dscn() ); }  // Using Sim as source of ib
    bool mod_vb() { return ( 1<<1 & modeling() || mod_vb_dscn() ); }  // Using Sim as source of vb
    bool mod_tb() { return ( 1<<0 & modeling() || mod_tb_dscn() ); }  // Using Sim as source of tb
    bool mod_none() { return ( 0==modeling() ); }                     // Using all

    // put
    void put_all_dynamic();
    void put_amp(const float input) { amp_p->check_set_put(input); }
    void put_booted(const bool input) { booted_p->check_set_put(input); }
    void get_booted() { booted_p->get(); }
    void put_cutback_gain_slr(const float input) { cutback_gain_slr_p->check_set_put(input); }
    void put_Debug(const int input) { debug_p->check_set_put(input); }
    void put_Delta_q(const double input) { delta_q_p->check_set_put(input); }
    void put_delta_q() {}
    void put_delta_q_model(const double input) { delta_q_model_p->check_set_put(input); }
    void put_delta_q_model() {}
    void put_Dw(const float input) { Dw_p->check_set_put(input); }
    void put_Freq(const float input) { freq_p->check_set_put(input); }
    void put_ib_bias_all(const float input) { ib_bias_all_p->check_set_put(input); }
    void put_ib_bias_amp(const float input) { ib_bias_amp_p->check_set_put(input); }
    void put_ib_bias_noa(const float input) { ib_bias_noa_p->check_set_put(input); }
    void put_ib_disch_slr(const float input) { ib_disch_slr_p->check_set_put(input); }
    void put_ib_force(const int8_t input) { ib_force_p->check_set_put(input); }
    void put_Iflt(const int input) { iflt_p->check_set_put(input); }
    void put_Ihis(const int input) { ihis_p->check_set_put(input); }
    void put_Isum(const int input) { isum_p->check_set_put(input); }
    void put_Inj_bias(const float input) { inj_bias_p->check_set_put(input); }
    void put_Preserving(const uint8_t input) { preserving_p->check_set_put(input); }
    void put_Tb_bias_hdwe(const float input) { Tb_bias_hdwe_p->check_set_put(input); }
    void put_Time_now(const uint32_t input) { Time_now_p->check_set_put(input); }
    void put_Type(const uint8_t input) { Type_p->check_set_put(input); }
    void put_Vb_bias_hdwe(const float input) { Vb_bias_hdwe_p->check_set_put(input); }
    
    void put_modeling(const uint8_t input) { modeling_p->check_set_put(input); modeling_ = modeling();}
    void put_fault(const Flt_st input, const uint8_t i) { fault_[i].copy_to_Flt_ram_from(input); }
    //
    Flt_st put_history(const Flt_st input, const uint8_t i);
    bool tweak_test() { return ( 1<<3 & modeling() ); } // Driving signal injection completely using software inj_bias 
    FloatV *amp_p;
    BooleanV *booted_p;
    FloatV *cutback_gain_slr_p;
    IntV *debug_p;
    DoubleV *delta_q_p;
    DoubleV *delta_q_model_p;
    FloatV *Dw_p;
    FloatV *freq_p;
    FloatV *ib_bias_all_p;
    FloatV *ib_bias_amp_p;
    FloatV *ib_bias_noa_p;
    FloatV *ib_disch_slr_p;
    Int8tV *ib_force_p;
    Uint16tV *iflt_p;
    Uint16tV *ihis_p;
    FloatV *inj_bias_p;
    Uint16tV *isum_p;
    Uint8tV *modeling_p;
    Uint8tV *preserving_p;
    FloatV *Tb_bias_hdwe_p;
    ULongV *Time_now_p;
    Uint8tV *Type_p;
    FloatV *Vb_bias_hdwe_p;
    FloatV *vsat_add_p;

protected:

    SerialRAM *rP_;
    Flt_st *fault_;
    Flt_st *history_;
    uint16_t next_;
    uint16_t nflt_;         // Length of Flt_ram array for fault snapshot
    uint16_t nhis_;         // Length of Flt_ram array for fault history
    uint16_t nsum_;         // Length of Sum array for history

    float amp_;
    bool booted_;                // True if device has been bootstrapped (i.e. initialized)
    float cutback_gain_slr_;
    float Dw_;
    int debug_;
    double delta_q_;
    double delta_q_model_;
    float freq_;
    float ib_bias_all_;
    float ib_bias_amp_;
    float ib_bias_noa_;
    float ib_disch_slr_;
    int8_t ib_force_;
    uint16_t iflt_;
    uint16_t ihis_;
    float inj_bias_;
    uint16_t isum_;
    uint8_t modeling_;
    uint8_t preserving_;
    float Tb_bias_hdwe_;
    uint32_t Time_now_;
    uint8_t type_;
    float Vb_bias_hdwe_;
    float vsat_add_;             // Saturation voltage bias, V

};
