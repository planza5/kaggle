import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold

import load_module as lm

covariates = [
    "donor_related_Related",
    "prim_disease_hct_NHL",
    "hepatic_mild",
    "melphalan_dose_missing",
    "sex_match_missing",
    "hla_low_res_6",
    "prim_disease_hct_MPN",
    "ethnicity",
    "hla_match_dqb1_low",
    "hla_match_c_low",
    "hla_match_dqb1_high",
    "tbi_status_TBI +- Other -cGy fractionated",
    "tce_match",
    "rituximab",
    "hla_match_c_high_1.0",
    "hla_match_a_low",
    "prior_tumor",
    "melphalan_dose_N/A Mel not given",
    "prim_disease_hct_Other acute leukemia",
    "prim_disease_hct_ALL",
    "tbi_status_TBI +- Other -cGy unknown dose",
    "prim_disease_hct_AML",
    "tce_imm_match",
    "prim_disease_hct_Solid tumor",
    "prim_disease_hct_IMD",
    "hla_match_c_high_2.0",
    "arrhythmia",
    "donor_related_Unrelated",
    "tbi_status_TBI +- Other -cGy single",
    "obesity",
    "sex_match_M-M",
    "tbi_status_TBI +- Other >cGy",
    "prim_disease_hct_CML",
    "prim_disease_hct_MDS",
    "tbi_status_TBI +- Other unknown dose",
    "hla_low_res_10",
    "donor_related_missing",
    "prim_disease_hct_IPA",
    "prim_disease_hct_SAA",
    "hla_high_res_8",
    "donor_age",
    "renal_issue",
    "hla_low_res_8",
    "peptic_ulcer",
    "hla_match_b_high",
    "hla_high_res_6",
    "hla_high_res_10",
    "vent_hist",
    "hla_match_drb1_low",
    "graft_type",
    "prod_type",
    "prim_disease_hct_Other leukemia",
    "prim_disease_hct_HD",
    "hla_nmdp_6",
    "conditioning_intensity",
    "hla_match_a_high",
    "psych_disturb",
    "rheum_issue",
    "pulm_moderate",
    "hla_match_b_low",
    "prim_disease_hct_PCD",
    "tce_div_match",
    "tbi_status_TBI + Cy +- Other",
    "in_vivo_tcd",
    "diabetes",
    "hepatic_severe",
    "cmv_status",
    "cyto_score_detail",
    "tbi_status_TBI +- Other <=cGy",
    "prim_disease_hct_HIS",
    "cyto_score",
    "prim_disease_hct_IEA",
    "prim_disease_hct_IIS",
    "age_at_hct",
    "karnofsky_score",
    "sex_match_M-F",
    "pulm_severe",
    "cardiac",
    "dri_score",
    "gvhd_proph",
    "sex_match_F-M",
    "year_hct",
    "comorbidity_score"
]

def summary(df):
    cph = CoxPHFitter()
    cph.fit(df, duration_col='efs_time',
            event_col='efs')

    cph.print_summary()

def cross_validation_cox(df, duration_col, event_col, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    c_indexes = []
    for train_index, test_index in kf.split(df):
        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]
        cph = CoxPHFitter()
        cph.fit(df_train, duration_col=duration_col, event_col=event_col)
        test_scores = cph.predict_partial_hazard(df_test)
        c_index = concordance_index(df_test[duration_col], -test_scores, df_test[event_col])
        c_indexes.append(c_index)
    c_index_promedio = np.mean(c_indexes)

    return c_index_promedio






def main():
    df_data, df_submit = lm.get_dfs()
    df_data = df_data.drop(['ID', 'race_group', 'mrd_hct', 'hla_match_drb1_high'],axis=1).copy()
    print('Init...')
    last_cindex = cross_validation_cox(df_data,'efs_time','efs')

    for feature in covariates:
        if feature not in df_data.columns:
            continue

        actual_cindex = cross_validation_cox(df_data.drop(feature,axis=1),'efs_time','efs')

        if actual_cindex>last_cindex:
            print('dropping '+feature+' improving cindex. actual='+str(actual_cindex)+' > last='+str(last_cindex))
            last_cindex = actual_cindex
            df_data = df_data.drop(feature,axis=1)
        else:
            print('dropping '+feature+' not improving cindex. last='+str(last_cindex)+'>= actual'+str(actual_cindex)+'...next feature')
            print(str(last_cindex)+'>'+str(actual_cindex))

    print(df_data.columns)


if __name__ == "__main__":
    main()