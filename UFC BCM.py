import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

import xgboost as xgb

pd.set_option("display.max_columns",500)
pd.set_option("display.max_rows",500)
pd.set_option("display.width",1000)

##########################################################################################################################################

fighters = pd.read_csv("raw_fighters.csv")

fighters["Fighter_Id"] = fighters.index.astype(str) #creating unique fight IDs derived from row indexes
fighters["Fighter_Id"] = (np.char.multiply("0",4-fighters["Fighter_Id"].str.len()))+fighters["Fighter_Id"] #filling ID with leading zero so they are all the same length

fighters["First"] = fighters["First"].fillna(" ")
fighters["Name"] = fighters["First"]+" "+fighters["Last"] #combining first and last name as they are in one column in the df: fights
fighters["Name"] = fighters["Name"].str.strip().str.lower() #given that some elements in "First" column are empty and might be whitespace
 
fighters = fighters.drop(["First","Last","Nickname","Belt","Wt.","W","L","D"],axis=1) #fighters weight and record are different for each fight so will be misleading

valid_stances = ["Orthodox","Southpaw","Switch","Other"]
fighters.loc[~(fighters["Stance"].isin(valid_stances)),"Stance"] = "Other" #removing redundant stances and replacing with "Other"

fighters["Ht._missing"] = (fighters["Ht."] == "--").astype(int) #adding a flag to indicate that height was imputated (missingness indicator)
height_parts = fighters["Ht."].str.extract(r"(\d+)'\s(\d+)") #using regex to split string into groups
feet = height_parts[0]
inches = height_parts[1]
fighters["Ht."] = ((feet.astype(float)*12)+inches.astype(float))*2.54 #converting height from inches in string to cm in float
fighters["Ht."] = fighters["Ht."].fillna(fighters["Ht."].median()) #impunating using median height - reduces the bias and better than removing data

fighters["Reach_missing"] = (fighters["Reach"] == "--").astype(int) #repeating for the reach column
reach_parts = fighters["Reach"].str.extract(r'(\d+).(\d+)"')
fighters["Reach"] = reach_parts[0].astype(float) + (reach_parts[1].astype(float)/100)
fighters["Reach"] = fighters["Reach"].fillna(fighters["Reach"].median())

fighter_key = ["Fighter_Id","Name"]
fighter_other = [col for col in fighters.columns if col not in fighter_key] #adding all columns that are in df but not in key_columns
new_order = fighter_key + fighter_other
fighters = fighters[new_order] #re ordering the columns so the key columns are left most

#########################################################################################################################################

fights = pd.read_csv("raw_fights_detailed.csv")
fights = fights.drop(["Referee","Method Details","Event_Id_y","Win/No Contest/Draw", 'KD_1', 'KD_2', 'STR_1', 'STR_2', 'TD_1', 'TD_2', 'SUB_1', 'SUB_2'],axis=1) 

print(len(fights))

events = pd.read_csv("raw_events.csv")
events = events.drop(["Name","Location"],axis=1)

fights = pd.merge(fights,events,left_on = "Event_Id_x",right_on = "Event_Id") #merging on event ID

fights["Date"] = pd.to_datetime(fights["Date"])
fights = fights.sort_values(by="Date").reset_index(drop=True) # reversing order of table so oldest fight at the top

fights["Method"] = fights["Method"].str.split(" ").str[0] #only taking first part of method description (SUB,TKO,U-DEC etc)

fights["Fight_Time"] = fights["Fight_Time"].astype(str)
fight_time_parts = fights["Fight_Time"].str.extract(r"(\d+):(\d+)")
mins = fight_time_parts.astype(float)[0]
secs = fight_time_parts.astype(float)[1]
fights["Fight_Time"] = mins + (secs/60)

fights["Ctrl_1"] = fights["Ctrl_1"].astype(str) 
Ctrl_1_parts = fights["Ctrl_1"].str.extract(r"(\d+):(\d+)")
mins = Ctrl_1_parts.astype(float)[0]
secs = Ctrl_1_parts.astype(float)[1]
fights["Ctrl_1"] = mins + (secs/60)

fights["Ctrl_2"] = fights["Ctrl_2"].astype(str)
Ctrl_1_parts = fights["Ctrl_2"].str.extract(r"(\d+):(\d+)")
mins = Ctrl_1_parts.astype(float)[0]
secs = Ctrl_1_parts.astype(float)[1]
fights["Ctrl_2"] = mins + (secs/60)

time_formats = ["3 Rnd (5-5-5)","5 Rnd (5-5-5-5-5)"]
fights["Time_Format"] = fights["Time Format"] #renaming to keep naming consistent
fights = fights.drop(["Time Format"],axis=1)
fights.loc[~(fights["Time_Format"].isin(time_formats)),"Time_Format"] = "Other"

stats = ["Kd_1", "Kd_2", "Sig. Str._1", "Sig. Str._2", "Sig. Str. %_1", "Sig. Str. %_2", "Total Str._1", "Total Str._2", "Td_1", "Td_2", "Td %_1", "Td %_2", "Sub. Att_1", "Sub. Att_2", "Rev._1", "Rev._2", "Ctrl_1", "Ctrl_2", "Head_1", "Head_2", "Body_1", "Body_2", "Leg_1", "Leg_2", "Distance_1", "Distance_2", "Clinch_1", "Clinch_2", "Ground_1", "Ground_2"]
totals_to_convert = ["Sig. Str._1", "Sig. Str._2", "Total Str._1", "Total Str._2", "Td_1", "Td_2", "Head_1", "Head_2", "Body_1", "Body_2", "Leg_1", "Leg_2", "Distance_1", "Distance_2", "Clinch_1", "Clinch_2", "Ground_1", "Ground_2"]
percentages_to_convert = ["Sig. Str. %_1", "Sig. Str. %_2", "Td %_1", "Td %_2"] 

for i in totals_to_convert: #converting totals in to total attempted and total successful instead of leaving as "n of m"
    current_parts = fights[i].str.extract(r"(\d+)\s+of\s+(\d+)")
    attempted = current_parts[1].astype(float)
    successful = current_parts[0].astype(float)
    fights[i] = successful
    attempted_name = "Attempted"+" "+i
    stats.append(attempted_name)
    fights[attempted_name] = attempted

for j in percentages_to_convert: #taking out percentage symbol and converting to a decimal between 0 and 1
    current_parts = fights[j].str.extract(r"(\d+)%")
    current_value = current_parts[0].astype(float)/100
    fights[j] = current_value

missing_flags = fights[stats].isna().astype(int) #vectorising impunation and addition of missing indicators to reduce fragemnation
missing_flags = missing_flags.rename(columns={col: col+"_missing" for col in missing_flags.columns.to_list()})
stats_medians = fights[stats].median()
fights[stats] = fights[stats].fillna(stats_medians)
fights = pd.concat([fights,missing_flags], axis=1)

fights = fights.rename(columns={"Sig. Str._1":"Sig_Str_1","Sig. Str._2": "Sig_Str_2", "Total Str._1": "Total_Str_1", "Total Str._2":"Total_Str_2"})
#########################################################################################################################################

fights["Fighter_1"] = fights["Fighter_1"].str.strip().str.lower()   #making sure formatting is the same as the fighters table
fights["Fighter_2"] = fights["Fighter_2"].str.strip().str.lower()

fighters.loc[fighters["Name"] == "montse rendon","Name"] = "montserrat rendon" #two fighters with inconsistent naming across tables so can just manually fix them
fights.loc[fights["Fighter_1"] == "montse rendon","Fighter_1"] = "montserrat rendon"
fights.loc[fights["Fighter_2"] == "montse rendon","Fighter_2"] = "montserrat rendon"

fights.loc[fights["Fighter_1"] == "jose medina","Fighter_1"] = "jose daniel medina"
fights.loc[fights["Fighter_2"] == "jose medina","Fighter_2"] = "jose daniel medina"

fighter_1 = fighters.copy() #taking a copy of fighters to merge with Fighter_1 in fights and then repeating for Fighter_2
new_cols = {col: col+"_1" for col in fighter_1.columns}
fighter_1 = fighter_1.rename(columns= new_cols)
fights = pd.merge(fights,fighter_1,left_on="Fighter_1",right_on="Name_1",how="left")
fights = fights.drop(columns = ["Name_1"])

fighter_2 = fighters.copy()
new_cols = {col: col+"_2" for col in fighter_2.columns}
fighter_2 = fighter_2.rename(columns= new_cols)
fights = pd.merge(fights,fighter_2,left_on="Fighter_2",right_on="Name_2",how="left")
fights = fights.drop(columns = ["Name_2"])

#########################################################################################################################################

#Ra and Rb are the elo ratings of fighters A and B
#Ea and Eb are the expected scores (probablity of winning) of fighters A and B
#Sa and Sb are the actual scores of fighters A and B (1 for a win, 0.5 for a draw and 0 for a loss)
# Sa+Sb=1 and Ea+Eb=1

def expected_score(Ra,Rb): #calculates the expected score of fighters
    Ea = 1/(1+math.pow(10,(Rb-Ra)/400))
    Eb = 1-Ea
    return Ea,Eb

def update_elo(Ra,Rb,Sa):
    K = 40 #this is the K-factor (max points a rating can go up by)
    Ea,Eb = expected_score(Ra,Rb)
    Ra_new = Ra + K*(Sa-Ea)
    Rb_new = Rb + K*((1-Sa)-Eb)
    return Ra_new,Rb_new

R_initial = 1500 #initial elo rating for all fighters
all_fighters = (pd.concat([fights["Fighter_1"],fights["Fighter_2"]])).unique() #list of names of all fighters
elo_ratings = {name: R_initial for name in all_fighters} #initalising elo for each fighter
record = {"W": 0, "D": 0, "L":0, "NC":0, "Total Fights":0.0} 
fighter_record = {name: record.copy() for name in all_fighters} #initialising each fighter's record
fe_stats = {"Total SS":0, "Total TD": 0, "SSpm":0, "TDpm":0, "Total KO":0, "Total SUB":0, "KO Rate":0, "SUB Rate":0, "Finish Rate":0, "Win Rate":0, "Total Fight Time":0.0}
fighter_fe_stats = {name: fe_stats.copy() for name in all_fighters} #initialising stats for each fighter

fights["Pre-Fight_Elo_1"] = 0.0 #creating columns for the feature engineered stats in fights
fights["Pre-Fight_Elo_2"] = 0.0 

fights["Elo_Diff"] = 0.0 #using difference in stats as a more efficient and direct indicator  
fights["Win_Rate_Diff"] = 0.0   
fights["SSpm_Diff"] = 0.0       
fights["TDpm_Diff"] = 0.0
fights["KO_Rate_Diff"] = 0.0
fights["SUB_Rate_Diff"] = 0.0
fights["Finish_Rate_Diff"] = 0.0

fights["Total_Fights_1"] = 0.0 #non-symmetrical so left separately
fights["Total_Fights_2"] = 0.0

fights["Height_Diff"] = 0.0
fights["Reach_Diff"] = 0.0

fights["Result_1_Binary"] = 0.0 #to be used as a target for the ML model
fights["Result_2_Binary"] = 0.0

stance_cols = ["Stance_1","Stance_2"]
encoded_stances = pd.get_dummies(fights[stance_cols],prefix=stance_cols,dtype=int)
fights = pd.concat([fights,encoded_stances],axis=1)
fights = fights.drop(columns=stance_cols)

for row in fights.itertuples(): #running through all the fights chronologically and updating stats for each fight
    index = row.Index
    fighter_1_name = row.Fighter_1 #getting fighter names
    fighter_2_name = row.Fighter_2
    elo_1 = elo_ratings.get(fighter_1_name) #getting "current" elo ratings from dictionary
    elo_2 = elo_ratings.get(fighter_2_name)
    elo_diff = elo_1 - elo_2
    fights.loc[index,"Elo_Diff"] = elo_diff #assigning elo difference and pre-fight elos
    fights.loc[index,"Pre-Fight_Elo_1"] = elo_1
    fights.loc[index,"Pre-Fight_Elo_2"] = elo_2
    result_1 = row.Result_1

    temp_record_1 = fighter_record.get(fighter_1_name, record.copy()) #taking copies of records to edit and update
    temp_record_2 = fighter_record.get(fighter_2_name, record.copy())
    temp_fe_stats_1 = fighter_fe_stats.get(fighter_1_name, fe_stats.copy()).copy()
    temp_fe_stats_2 = fighter_fe_stats.get(fighter_2_name, fe_stats.copy()).copy()

    fights.loc[index,"Height_Diff"] = fights.loc[index,"Ht._1"] - fights.loc[index,"Ht._2"]
    fights.loc[index,"Reach_Diff"] = fights.loc[index,"Reach_1"] - fights.loc[index,"Reach_2"]
    fights.loc[index,"Win_Rate_Diff"] = temp_fe_stats_1["Win Rate"] - temp_fe_stats_2["Win Rate"] #adding pre-fight stats to table using record data
    fights.loc[index,"SSpm_Diff"] = temp_fe_stats_1["SSpm"] - temp_fe_stats_2["SSpm"]
    fights.loc[index,"TDpm_Diff"] = temp_fe_stats_1["TDpm"] - temp_fe_stats_2["TDpm"]
    fights.loc[index,"KO_Rate_Diff"] = temp_fe_stats_1["KO Rate"] - temp_fe_stats_2["KO Rate"]
    fights.loc[index,"SUB_Rate_Diff"] = temp_fe_stats_1["SUB Rate"] - temp_fe_stats_2["SUB Rate"]
    fights.loc[index,"Finish_Rate_Diff"] = temp_fe_stats_1["Finish Rate"] - temp_fe_stats_2["Finish Rate"]
    fights.loc[index,"Total_Fights_1"] = temp_record_1["Total Fights"]
    fights.loc[index,"Total_Fights_2"] = temp_record_2["Total Fights"]


    if result_1 == "W": #determining score based on result column of fighter 1
        fights.loc[index,"Result_1_Binary"] = 1
        fights.loc[index,"Result_2_Binary"] = 0
        actual_score_1 = 1.0
        temp_record_1["W"] += 1
        temp_record_2["L"] += 1
        if row.Method == "KO/TKO":  #updating finish stats based on method of win
            temp_fe_stats_1["Total KO"] += 1
        elif row.Method == "SUB":
            temp_fe_stats_1["Total SUB"] += 1
    elif result_1 == "L":
        fights.loc[index,"Result_1_Binary"] = 0
        fights.loc[index,"Result_2_Binary"] = 1
        actual_score_1 = 0.0
        temp_record_1["L"] += 1
        temp_record_2["W"] += 1
        if row.Method == "KO/TKO":
            temp_fe_stats_2["Total KO"] += 1
        elif row.Method == "SUB":
            temp_fe_stats_2["Total SUB"] += 1
    else:
        fights.loc[index,"Result_1_Binary"] = 0
        fights.loc[index,"Result_1_Binary"] = 0
        actual_score_1 = 0.5
        if result_1 == "NC":
            temp_record_1["NC"] += 1
            temp_record_2["NC"] += 1
        else:
            temp_record_1["D"] += 1
            temp_record_2["D"] += 1

    temp_record_1["Total Fights"] += 1
    temp_record_2["Total Fights"] += 1 

    fighter_record[fighter_1_name] = temp_record_1
    fighter_record[fighter_2_name] = temp_record_2
    
    actual_score_2 = 1 - actual_score_1
    elo_1_new, elo_2_new = update_elo(elo_1,elo_2,actual_score_1) #calculating new elos after fight and assigning them in the dictionary
    elo_ratings[fighter_1_name] = elo_1_new
    elo_ratings[fighter_2_name] = elo_2_new
    
    ss_landed_1 = row.Sig_Str_1 #extracting current fights stats to update dicitonary for next fight
    ss_landed_2 = row.Sig_Str_2
    td_successful_1 = row.Td_1
    td_successful_2 = row.Td_2
    fight_time_current = row.Fight_Time
    
    decay_factor = 0.6 # 10% of the value is forgotten per fight meaning more emphasis on current form (temporal decay)
    temp_fe_stats_1["Total SS"] = (decay_factor * temp_fe_stats_1["Total SS"]) + ss_landed_1 
    temp_fe_stats_1["Total TD"] = (decay_factor * temp_fe_stats_1["Total TD"]) + td_successful_1
    temp_fe_stats_1["Total Fight Time"] = (decay_factor * temp_fe_stats_1["Total Fight Time"]) + fight_time_current
    temp_fe_stats_2["Total SS"] = (decay_factor * temp_fe_stats_2["Total SS"]) + ss_landed_2
    temp_fe_stats_2["Total TD"] = (decay_factor * temp_fe_stats_2["Total TD"]) + td_successful_2
    temp_fe_stats_2["Total Fight Time"] = (decay_factor * temp_fe_stats_2["Total Fight Time"]) + fight_time_current

    safe_time_1 = temp_fe_stats_1["Total Fight Time"] if temp_fe_stats_1["Total Fight Time"] > 0 else 0.01 #preventing a division by zero during debut fights
    safe_time_2 = temp_fe_stats_2["Total Fight Time"] if temp_fe_stats_2["Total Fight Time"] > 0 else 0.01
    safe_fights_1 = temp_record_1["Total Fights"] if temp_record_1["Total Fights"] > 0 else 1
    safe_fights_2 = temp_record_2["Total Fights"] if temp_record_2["Total Fights"] > 0 else 1

    temp_fe_stats_1["SSpm"] = temp_fe_stats_1["Total SS"]/safe_time_1 #updating more stats post-fight so they're ready to use in the next loop iteration
    temp_fe_stats_2["SSpm"] = temp_fe_stats_2["Total SS"]/safe_time_2
    temp_fe_stats_1["TDpm"] = temp_fe_stats_1["Total TD"]/safe_time_1
    temp_fe_stats_2["TDpm"] = temp_fe_stats_2["Total TD"]/safe_time_2
    temp_fe_stats_1["KO Rate"] = temp_fe_stats_1["Total KO"]/safe_fights_1
    temp_fe_stats_2["KO Rate"] = temp_fe_stats_2["Total KO"]/safe_fights_2
    temp_fe_stats_1["SUB Rate"] = temp_fe_stats_1["Total SUB"]/safe_fights_1
    temp_fe_stats_2["SUB Rate"] = temp_fe_stats_2["Total SUB"]/safe_fights_2
    temp_fe_stats_1["Finish Rate"] = (temp_fe_stats_1["Total KO"]+temp_fe_stats_1["Total SUB"])/safe_fights_1
    temp_fe_stats_2["Finish Rate"] = (temp_fe_stats_2["Total KO"]+temp_fe_stats_2["Total SUB"])/safe_fights_2
    temp_fe_stats_1["Win Rate"] = temp_record_1["W"]/safe_fights_1
    temp_fe_stats_2["Win Rate"] = temp_record_2["W"]/safe_fights_2

    fighter_fe_stats[fighter_1_name] = temp_fe_stats_1 #assigning the updated records back to the fighter's stats record
    fighter_fe_stats[fighter_2_name] = temp_fe_stats_2 

# Y will be the target (the thing we're trying to predict) which in this case is the "Result_1_Binary" column
# X will be the features (the data used to predit)

Y = fights["Result_1_Binary"]
cols_to_keep = ['Pre-Fight_Elo_1', 'Pre-Fight_Elo_2', 'Elo_Diff', 'Win_Rate_Diff', 'SSpm_Diff', 'TDpm_Diff', 'KO_Rate_Diff', 'SUB_Rate_Diff', 'Finish_Rate_Diff', 'Total_Fights_1', 'Total_Fights_2', 'Height_Diff', 'Reach_Diff','Stance_1_Orthodox', 'Stance_1_Other', 'Stance_1_Southpaw', 'Stance_1_Switch', 'Stance_2_Orthodox', 'Stance_2_Other', 'Stance_2_Southpaw', 'Stance_2_Switch', 'Ht._1','Ht._2','Reach_1','Reach_2']
X = fights[cols_to_keep]

split_index = int(0.8*len(X)) # using first 80% of fights to train data and using the most recent 20% as test data
Y_training = Y.iloc[0:split_index]
Y_testing = Y.iloc[split_index:]
X_training = X.iloc[0:split_index,:] #splitting up the data into training and testing subsets
X_testing = X.iloc[split_index:,:]

cols_to_swap = [("Pre-Fight_Elo_1", "Pre-Fight_Elo_2"), ('Total_Fights_1', 'Total_Fights_2'), ('Stance_1_Orthodox', 'Stance_2_Orthodox'),('Stance_1_Other', 'Stance_2_Other'), ('Stance_1_Southpaw', 'Stance_2_Southpaw'), ('Stance_1_Switch', 'Stance_2_Switch'), ('Ht._1','Ht._2'), ('Reach_1','Reach_2')] 
cols_to_flip = ['Elo_Diff', 'Win_Rate_Diff', 'SSpm_Diff', 'TDpm_Diff', 'KO_Rate_Diff', 'SUB_Rate_Diff', 'Finish_Rate_Diff', 'Height_Diff', 'Reach_Diff']
X_flipped = X_training.copy()
for col_A, col_B in cols_to_swap:
    X_flipped[col_A], X_flipped[col_B] = X_flipped[col_B].copy(), X_flipped[col_A].copy()
X_flipped[cols_to_flip] *= -1
Y_flipped = fights.loc[X_training.index, "Result_2_Binary"].copy()
X_training_augmented = pd.concat([X_training,X_flipped],ignore_index=True) #swapping around fighter_1 and fighter_2 so model learns both wins and losses (result_1_binary contains mostly wins so augmenting data provides grater exposure to losses)
Y_training_augmented = pd.concat([Y_training,Y_flipped],ignore_index=True)

scaler = StandardScaler()
X_training_scaled = scaler.fit_transform(X_training_augmented) #scaling the data so that the mean is zero and std is one (ensures each column is initially equally weighted)
X_testing_scaled = scaler.transform(X_testing)

rf_model = RandomForestClassifier(n_estimators=600, random_state=42, class_weight="balanced") #creating the random forest model and fitting it to the data
#class weight balanced to ignore the frequency of values in a column (most of Result_1_Binary = 1.0 so is unfairly weighted)

rf_model.fit(X_training_scaled, Y_training_augmented)

Y_prediction = rf_model.predict(X_testing_scaled) #prediciting using the test data

print("Classification Report")
print(classification_report(Y_testing, Y_prediction))
print(" ")
print("Accuracy Score")
print(accuracy_score(Y_testing, Y_prediction))

importances = rf_model.feature_importances_
feature_names = X_training_augmented.columns 
feature_importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)
print(" ")
print(feature_importances_df.head(len(cols_to_keep)))