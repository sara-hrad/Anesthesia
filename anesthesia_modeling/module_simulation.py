import numpy as np
import control as ct


def cc2ss(clearance, volume):       # create the state-space format of PK model
    n = len(clearance)
    k1 = clearance / volume[0]
    k2 = clearance[1:] / volume[1:]
    a = np.vstack((np.hstack((-np.sum(k1), k1[1:])), np.hstack((np.transpose(k2)[:, None], -np.diag(k2)))))
    b = np.vstack(([1 / volume[0]], np.zeros((n - 1, 1))))
    c = np.array([[1, 0, 0]])
    d = np.array([[0]])
    a = a / 60
    sys = ct.ss(a, b, c, d)
    return sys


def eleveld_pk_model(gender, age, wgt, height):
    # Covariates
    bmi = wgt / (height ** 2)  # Body Mass Index
    pma = age * 365.25 / 7 + 40  # Post Menstrual Age in weeks (Usually Age + 40 weeks)
    # Options
    opiates = 'on'  # in the presence of opiates (on) or absence (off)
    blood_sampling = 'arterial'  # Blood sampling site (arterial) vs (venous)
    # PK identified parameters
    theta1 = 6.28  # V1_ref [litre]
    theta2 = 25.5  # V2_ref [litre]
    theta3 = 273  # V3_ref [litre]
    theta4 = 1.79  # CL_ref (male) [Litre/min]
    theta5 = 1.75  # Q2_ref [Litre/min]
    theta6 = 1.11  # Q3_ref [Litre/min]
    theta7 = 0.191  # Typical residual error
    theta8 = 42.3  # CL maturation E50 [weeks]
    theta9 = 9.06  # CL maturation E50 slope
    theta10 = -0.0156  # Smaller V2 with age
    theta11 = -0.00286  # Lower CL with age
    theta12 = 33.6  # Weight for 50% of maximal V1 [kg]
    theta13 = -0.0138  # Smaller V3 with age
    theta14 = 68.3  # Maturation of Q3 [weeks]
    theta15 = 2.10  # CL_ref (female) [Litre/min]
    theta16 = 1.3  # Higher Q2 for maturation of Q3
    theta17 = 1.42  # V1 venous samples (children)
    theta18 = 0.68  # Higher Q2 venous samples
    # Reference Covariates
    gender_ref = 'male'  # Gender of the reference patient
    age_ref = 35  # Reference Age in [year]
    pma_ref = age_ref * 365.25 / 7 + 40  # Reference Post Menstrual Age (Usually Age + 40 weeks)
    wgt_ref = 70  # Reference Weight in [kg]
    height_ref = 170  # Reference Height is centimeter
    bmi_ref = 24.2  # Reference Body Mass Index
    v2_ref = theta2  # V2_ref [litre]
    v3_ref = theta3  # V3_ref [litre]

    # PKPD Model
    # Intermediate Coefficients
    f_aging_theta10 = np.exp(theta10 * (age - age_ref))
    f_central_wgt = wgt / (wgt + theta12)
    f_central_wgt_ref = wgt_ref / (wgt_ref + theta12)
    f_cl_maturation = pma ** theta9 / (pma ** theta9 + theta8 ** theta9)
    f_cl_maturation_ref = pma_ref ** theta9 / (pma_ref ** theta9 + theta8 ** theta9)
    f_q3maturation = (age * 365.25 / 7 + 40) / (age * 365.25 / 7 + 40 + theta14)
    f_q3maturation_ref = (age_ref * 365.25 / 7 + 40) / (age_ref * 365.25 / 7 + 40 + theta14)

    # The impact of gender on the intermediate coefficients
    if gender == 'male':
        f_alsallami = (0.88 + (1 - 0.88) / (1 + (age / 13.4) ** (-12.7))) * (9270 * wgt / (6680 + 216 * bmi))
        theta_cl = theta4
    elif gender == 'female':
        f_alsallami = (1.11 + (1 - 1.11) / (1 + (age / 7.1) ** (-1.1))) * (9270 * wgt / (8780 + 244 * bmi))
        theta_cl = theta15

    f_alsallami_ref = ((0.88 + (1-0.88) / (1 + (age_ref / 13.4) ** (-12.7)))*(9270 * wgt_ref / (6680 + 216 * bmi_ref)))
    # The impact of opiates' presence on the intermediate coefficients
    if opiates == 'on':
        f_opiates_theta13 = np.exp(theta13 * age)
        f_opiates_theta11 = np.exp(theta11 * age)
    elif opiates == 'off':
        f_opiates_theta13 = 1
        f_opiates_theta11 = 1

    # Volume of compartments [Litre]
    v1_arterial = theta1 * f_central_wgt / f_central_wgt_ref
    v1_venous = v1_arterial * (1 + theta17 * (1 - f_central_wgt))
    v2 = theta2 * (wgt / wgt_ref) * f_aging_theta10
    v3 = theta3 * (f_alsallami / f_alsallami_ref) * f_opiates_theta13

    # Elimination and inter-compartment clearance rates [litre/min]
    cl = theta_cl * (wgt / wgt_ref) ** 0.75 * f_cl_maturation * f_opiates_theta11 / f_cl_maturation_ref
    q2_arterial = theta5 * (v2 / v2_ref) ** 0.75 * (1 + theta16 * (1 - f_q3maturation))
    q2_venous = q2_arterial * theta18
    q3 = theta6 * (v3 / v3_ref) ** 0.75 * f_q3maturation / f_q3maturation_ref

    # The effect of the blood sampling site on the volume and
    if blood_sampling == 'arterial':
        v1 = v1_arterial
        q2 = q2_arterial
    elif blood_sampling == 'venous':
        v1 = v1_venous
        q2 = q2_venous

    volume = np.array([v1, v2, v3])
    clearance = np.array([cl, q2, q3])
    sys = cc2ss(clearance, volume)
    return sys


def pd_linear_model(ke0, ce50, t_d):
    num = np.array([ke0/(2*ce50*60)])
    den = np.array([1, ke0/60])
    sys = ct.tf(num, den)
    time_delay_pad_app = ct.tf(ct.pade(t_d)[0], ct.pade(t_d)[1] )
    sys = ct.series(sys, time_delay_pad_app)
    return sys


def pd_model_hillfunction(ce, e0, gamma):
    e0 = (100 - e0)/100
    ce = ce/(1-e0)
    ce[ce<0] = 0
    e = e0 + (1-e0)*ce**gamma/(0.5**gamma + ce**gamma)
    return e


def gm_filter():
    af = np.array([[0.875000000000001, -0.882496902584599], [1.13314845306682, -1.12499999999999]])  # 30 sec IIR filter
    bf = np.array([[0.0678724069605523], [-0.0737711496728008]])
    cf = np.array([0.154464292547276, -0.0448027554524884])
    df = 0.007190984592330
    gns = ct.ss(af, bf, cf, df)
    return gns
