
import numpy as np
import pandas as pd
import pathlib
from pathlib import Path
import os
import glob
import datetime


ncaa = pd.read_csv("ncaa.csv")

# Question 1

# Find all schools that have lost the championship more than one time. Report 
# your results in a Series that has the schools as index and number of 
# championships lost for values, sorted alphabetically by school.

championships = ncaa.loc[ncaa["Round"] == 6]
home_ls = championships.loc[championships["Score"] < championships["Score.1"]][["Year", "Team"]].rename(columns = {"Team":"Loser"})
away_ls = championships.loc[championships["Score"] > championships["Score.1"]][["Year", "Team.1"]].rename(columns = {"Team.1":"Loser"})
losers = pd.concat([home_ls, away_ls])
l_counts = losers.groupby("Loser").count()
q1 = l_counts.loc[l_counts["Year"] > 1].squeeze()


# Question 2

# Determine all schools that have progressed to the Sweet-16 of the tournament
# at least 15 times.  (The Sweet-16 is the 3rd round of the toournament.)
# Report your results as a Series with schools as index and number of times
# in the Sweet-16 as values, sorted by values from largest to smallest.

sweet16 = ncaa.loc[ncaa["Round"] == 3]
counts_s16h = sweet16.groupby("Team")["Year"].count().reset_index()[["Year", "Team"]]
counts_s16a = sweet16.groupby("Team.1")["Year"].count().reset_index()[["Year", "Team.1"]].rename(columns = {"Team.1":"Team"})
sweet_teams = pd.merge(counts_s16h, counts_s16a, on="Team")
sweet_teams["total"] = sweet_teams["Year_x"] + sweet_teams["Year_y"]
fifteen_apps = sweet_teams.loc[sweet_teams["total"] >= 15][["Team", "total"]].sort_values(by="total", ascending=False)
q2 = fifteen_apps.set_index("Team").squeeze()



# Question 3

# Find all years when the school that won the tournament was seeded 
# 3 or lower. (Remember that "lower" seed means a bigger number!) Give  
# a DataFrame with years as index and corresponding school and seed
# as columns (from left to right).  Sort by year from least to most recent.

home_winners = championships.loc[championships["Score"] > championships["Score.1"]][["Year", "Team", "Seed"]].rename(columns = {"Team":"Team"})
away_winners = championships.loc[championships["Score"] < championships["Score.1"]][["Year", "Team.1", "Seed.1"]].rename(columns = {"Team.1":"Team", "Seed.1":"Seed"})
winners = pd.concat([home_winners, away_winners]).sort_values("Year").set_index("Year")
q3 = winners.loc[winners["Seed"] >= 3]


# Question 4

# Determine the average tournament seed for each school.  Make a Series
# of all schools that have an average seed of 5.0 or higher (that is,
# the average seed number is <= 5.0).  The Series should have schools
# as index and average seeds as values, sorted alphabetically by
# school

R1h = ncaa.loc[ncaa["Round"] == 1][["Team", "Seed"]]
R1a = ncaa.loc[ncaa["Round"] == 1][["Team.1", "Seed.1"]].rename(columns = {"Team.1":"Team", "Seed.1":"Seed"})
Teams = pd.concat([R1h, R1a])
grouped = Teams.groupby("Team").mean()
q4 = grouped.loc[grouped["Seed"] <= 5].squeeze()


# Question 5

# For each tournament round, determine the percentage of wins by the
# higher seeded team. (Ignore games of teams with the same seed.)
# Give a Series with round number as index and percentage of wins
# by higher seed as values sorted by round in order 1, 2, ..., 6. 
# (Remember, a higher seed means a lower seed number.)

ncaa["which_seed"] = ncaa["Seed"]-ncaa["Seed.1"]
ncaa["score_difference"] = ncaa["Score"] - ncaa["Score.1"]
upsets = ncaa.loc[((ncaa["which_seed"] < 0) & (ncaa["score_difference"] < 0)) | ((ncaa["which_seed"] > 0) & (ncaa["score_difference"] > 0))]
num_games = ncaa.groupby("Round")["Team"].count()
upsets_by_round = upsets.groupby("Round")["Team"].count()
q5 = upsets_by_round/num_games


# Question 6

# For each seed 1, 2, 3, ..., 16, determine the average number of games
# won per tournament by a team with that seed.  Give a Series with seed 
# number as index and average number of wins as values, sorted by seed 
# number 1, 2, 3, ..., 16. (Hint: There are 35 tournaments in the data set
# and each tournament starts with 4 teams of each seed.  We are not 
# including "play-in" games which are not part of the data set.)

home_games = ncaa.loc[ncaa["score_difference"] > 0]["Seed"]
away_games = ncaa.loc[ncaa["score_difference"] < 0]["Seed.1"].rename({"Seed.1":"Seed"})
total_games = pd.concat([home_games,away_games])
q6 = (total_games.value_counts()/(35)).sort_index()


# Question 7

# Are some schools particularly good at winning games as a lower seed?  For
# each team, determine the percentage of games won by that team when that team 
# was a lower seed than their opponent.  Give a Series of all schools that have 
# won more than 60% of their games while the lower seed, with school as index
# and percentage of victories as values, sorted by percentage from greatest 
# to least.

home_low_seed = ncaa.loc[ncaa["Seed"] > ncaa["Seed.1"]]["Team"]
away_low_seed = ncaa.loc[ncaa["Seed"] < ncaa["Seed.1"]]["Team.1"].rename({"Team.1":"Team"})
total_low_seed = pd.concat([home_low_seed,away_low_seed]).value_counts().reset_index().rename(columns={"index":"Winner"})


home_upsets = ncaa.loc[((ncaa["which_seed"] < 0) & (ncaa["score_difference"] < 0))][["Year", "Team.1"]].rename(columns = {"Team.1":"Winner"})
away_upsets = ncaa.loc[((ncaa["which_seed"] > 0) & (ncaa["score_difference"] > 0))][["Year", "Team"]].rename(columns = {"Team":"Winner"})
all_upsets = pd.concat([home_upsets,away_upsets])
upsets_by_team = all_upsets.groupby("Winner").count().reset_index()
team_low = pd.merge(total_low_seed,upsets_by_team,on="Winner")
team_low["prop_upsets"] = (team_low["Year"] / team_low[0])*100
q7 = team_low.loc[team_low["prop_upsets"] > 60].sort_values("prop_upsets", ascending = False).set_index("Winner")["prop_upsets"]


# Question 8

# Is there a region that consistently has the closest games?  For each year,
# determine which region has the lowest average point differential (winning 
# minus losing score), ignoring the games in 'Final Four' and 'Championship'.
# Give you answer as a data frame, with year as index, a column with the
# name of the region that has the lowest average, and a column with the average
# (in that order). Note that the names of the four regions are not always the
# same.

ncaa["abs_diff"] = abs(ncaa["score_difference"])
no_champ4 = ncaa.loc[(ncaa["Region Name"] != "Championship") & (ncaa["Region Name"] != "Final Four")]
region_diff = no_champ4.groupby(["Year","Region Name"])["abs_diff"].mean().reset_index()
low_years = region_diff.groupby("Year").min("abs_diff").reset_index()
q8 = pd.merge(region_diff,low_years,on=["abs_diff","Year"]).set_index("Year")



# Question 9

# For each year's champion, determine their average margin of victory 
# across all of their games in that year's tournament. Find the champions
# that have an average margin of victory of no more than 10. Give a DataFrame 
# with year as index and champion and average margin of victory as columns
# (from left to right), sorted by from lowest to highest average victory 
# margin.

away_champs = championships.loc[championships["Score"] < championships["Score.1"]][["Year", "Team.1"]].rename(columns = {"Team.1":"Winner"})
home_champs = championships.loc[championships["Score"] > championships["Score.1"]][["Year", "Team"]].rename(columns = {"Team":"Winner"})
champions = pd.concat([away_champs,home_champs]).sort_values("Year")
ncaa9 = pd.merge(ncaa,champions,on="Year")
champ_games_played = ncaa9.loc[(ncaa9["Team"] == ncaa9["Winner"]) | (ncaa9["Team.1"] == ncaa9["Winner"])]
avg_mov = champ_games_played.groupby(["Year","Winner"])["abs_diff"].mean().reset_index().set_index("Year").sort_values("abs_diff", ascending = True).rename(columns = {"Winner":"Team","abs_diff":"Victory Margin"})
q9 = avg_mov.loc[avg_mov["Victory Margin"] <= 10]


# Question 10

# Determine the 2019 champion.  Use code to extract the correct school,
# not your knowledge of college backetball history.

UVA = championships.loc[championships["Year"] == 2019]

q10 = UVA.iloc[0,6]












