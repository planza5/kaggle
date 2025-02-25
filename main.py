import main_rsf
import main_rsf as m
def go():
    no_train=['ID','race_group','efs','efs_time']
    df_train , _  =m.get_dfs()
    rsf = m.create_model()
    rsf = m.train(rsf,df_train)

    scores = main_rsf.evaluate_fairness_by_race(
        rsf,
        df_train.drop(no_train,axis=1),
        durations=df_train['efs_time'],
        race_groups=df_train['race_group'],
        events=df_train['efs']
    )

if __name__ == "__main__":
    go()

