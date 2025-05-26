# Load libraries
install.packages("caret")
install.packages("sentimentr")
install.packages("tidytext")
install.packages("stringr")
install.packages("text2vec")
install.packages("textTinyR")
install.packages("text")
install.packages("rvest")    # for web scraping
install.packages("FSelector")
install.packages("rJava")
install.packages("DMwR")
install.packages("smotefamily")


library(caret)
library(syuzhet)
library(sentimentr)
library(tidyverse)
library(rpart)
library(tidytext)
library(dplyr)
library(stringr)
library(text2vec)
library(reshape2)
library(proxy)
library(textTinyR)
library(text)
library(e1071)  # For skewness function
library(rvest)
library(rJava)
library(DMwR)
library(smotefamily)

# Load data files
train_x <- read_csv("ks_training_X.csv", show_col_types = FALSE)
train_y <- read_csv("ks_training_y.csv", show_col_types = FALSE)
test_x <- read_csv("ks_test_X.csv", show_col_types = FALSE)
test_y <- read_csv("ks_test_y_FAKE.csv", show_col_types = FALSE)

# Join the training y to the training x file
train_success <- train_x %>%
  left_join(train_y, by = "id") %>%
  mutate(original_set = "tr")

test_success <- test_x %>%
  left_join(test_y, by = "id") %>%
  mutate(original_set = "te")

# Stack the training and test data
all_data <- rbind(train_success, test_success)

############################Feature Engineering#################################
# NILAY


all_data <- all_data %>%
  mutate(launch_weekday = as.factor(ifelse(
    weekdays(launched_at) %in% c('Monday', 'Tuesday'), 'Beginning',
    ifelse(weekdays(launched_at) %in% c('Wednesday', 'Thursday', 'Friday'), 'MidWeek', 'Weekend'))
  ))

#tag list
tag_list <- unlist(strsplit(as.character(all_data$tag_names), "\\|"))
tag_freq <- table(tag_list)


# Convert columns to datetime format
all_data <- all_data %>%
  mutate(
    launched_at = as.POSIXct(launched_at, format = "%Y-%m-%d %H:%M:%S"),
    created_at = as.POSIXct(created_at, format = "%Y-%m-%d %H:%M:%S"),
    deadline = as.POSIXct(deadline, format = "%Y-%m-%d %H:%M:%S")
  )


all_data <- all_data %>%
  mutate(
    log_goal = log(goal),
    time_to_launch = as.numeric(difftime(launched_at, created_at, units = "days")),
    project_duration = as.numeric(difftime(deadline, launched_at, units = "days")),
    agediff_project = maxage_project - minage_project,
    tag_count = sapply(strsplit(all_data$tag_names, "|"), length),
    more_avg_sentence = ifelse(sentence_counter>avgsentencelength, 1, 0), 
    words_per_sentence = ifelse(sentence_counter!=0,num_words/sentence_counter,0),
    verbs_per_sentence = ifelse(sentence_counter!=0,VERB/sentence_counter,0),
    adj_per_sentence = ifelse(sentence_counter!=0,ADJ/sentence_counter,0),
    noun_per_sentence = ifelse(sentence_counter!=0,ADJ/sentence_counter,0),
    male_dominant_project = ifelse(male_project>female_project,1,0),
    smiling_project_flag = ifelse(smiling_project>=0.5,1,0),
    any_pic = ifelse(pmax(isTextPic, isLogoPic, isCalendarPic, isDiagramPic, isShapePic, na.rm = TRUE) == 1, 1, 0),
    is_Town = as.factor(ifelse(location_type == 'Town', 1, 0)),
    state_name = substr(location_slug, nchar(location_slug) - 1, nchar(location_slug)),
    interaction_goal_numwords = goal * num_words,              # Interaction between goal and num_words
    interaction_goal_avgwordlength = goal * avg_wordlengths,   # Interaction between goal and avg_wordlengths
    is_sentiment_positive = ifelse(afinn_pos > afinn_neg, 1, 0),
    tag_popularity = log(sapply(strsplit(as.character(tag_names), "\\|"), function(x) sum(tag_freq[x])))
  )


#all_data$blurb_sentiment <- sentiment_by(all_data$blurb)$ave_sentiment

# Town vs Others
# all_data <- all_data %>%
#   mutate(
#     location_type = case_when(
#     location_type %in% c('Town') ~ 'Town',
#     TRUE ~ 'Others'),
#     continent = as.factor(location_type)
#     )


all_data <- all_data %>%
  mutate(goal_by_avg_sentence = ifelse(sentence_counter!=0,goal/sentence_counter,0))




# Step 1: Define the list of attractive words
reward_attractive_words <- c("exclusive", "premium","unique","surprise","shoutout", "limited", "early access", "personalized", 
                             "thank you", "signed", "behind the scenes", 
                             "bundle", "special edition", "discount","dvd","cd","custom","recognition", "shoutout", "first")

# Step 2: Create the new column based on the presence of attractive words or numbers
all_data <- all_data %>%
  mutate(reward_desc_attractive_words = ifelse(
    str_detect(reward_descriptions, paste(reward_attractive_words, collapse = "|")),  # Check for attractive words
    1,  # If condition met, assign 1
    0   # Otherwise, assign 0
  ))

all_data <- all_data %>%
  mutate(reward_desc_attractive_word_count = str_count(reward_descriptions, 
                                                       paste(reward_attractive_words, collapse = "|")))


top_state_category_combinations <- list(
  "CA" = c("film & video", "technology", "design"),
  "NY" = c("art", "fashion", "music"),
  "TX" = c("games", "technology", "film & video"),
  "WA" = c("technology", "games"),
  "MA" = c("technology", "design"),
  "CO" = c("crafts", "film & video"),
  "OR" = c("food", "design"),
  "IL" = c("film & video", "publishing"),
  "GA" = c("film & video", "technology"),
  "NC" = c("technology", "art"),
  "MI" = c("games", "publishing"),
  "TN" = c("music", "film & video"),
  "FL" = c("games", "film & video")
)

# Create a new column 'top_state_category_flag' based on the combination of 'state_name' and 'category_parent'
all_data <- all_data %>%
  mutate(top_state_category_flag = ifelse(
    (state_name == "ca" & category_parent %in% top_state_category_combinations[["CA"]]) |
      (state_name == "ny" & category_parent %in% top_state_category_combinations[["NY"]]) |
      (state_name == "tx" & category_parent %in% top_state_category_combinations[["TX"]]) |
      (state_name == "wa" & category_parent %in% top_state_category_combinations[["WA"]]) |
      (state_name == "ma" & category_parent %in% top_state_category_combinations[["MA"]]) |
      (state_name == "co" & category_parent %in% top_state_category_combinations[["CO"]]) |
      (state_name == "or" & category_parent %in% top_state_category_combinations[["OR"]]) |
      (state_name == "il" & category_parent %in% top_state_category_combinations[["IL"]]) |
      (state_name == "ga" & category_parent %in% top_state_category_combinations[["GA"]]) |
      (state_name == "nc" & category_parent %in% top_state_category_combinations[["NC"]]) |
      (state_name == "mi" & category_parent %in% top_state_category_combinations[["MI"]]) |
      (state_name == "tn" & category_parent %in% top_state_category_combinations[["TN"]]) |
      (state_name == "fl" & category_parent %in% top_state_category_combinations[["FL"]]), 
    1, 0
  ))

all_data <- all_data %>%
  mutate(num_words_creator_name = str_count(creator_name, "\\w+"))

# Step 1: Define the thresholds for the categories
low_threshold <- 31   # 1st Quartile
medium_threshold <- 74  # 3rd Quartile

# Step 2: Create a new column for tag_count category
all_data <- all_data %>%
  mutate(tag_count_category = as.factor(case_when(
    tag_count <= low_threshold ~ "Low",
    tag_count > low_threshold & tag_count <= medium_threshold ~ "Medium",
    tag_count > medium_threshold ~ "High"
  )))

# View the new column
summary(all_data$tag_count_category)


awareness_data <- all_data %>%
  group_by(state_name) %>%
  summarise(
    total_projects = n(),  # Count of all projects
    total_funding = sum(as.numeric(unlist(strsplit(reward_amounts, ","))), na.rm = TRUE)  # Total funding raised
  ) %>%
  ungroup()

# Step 2: Create an awareness score
awareness_data <- awareness_data %>%
  mutate(awareness_score = total_projects * 0.4 + total_funding * 0.6)  # Adjust weights as necessary

# Step 3: Join the awareness score back to the main dataset
all_data <- all_data %>%
  left_join(awareness_data, by = "state_name")


all_data <- all_data %>%
  mutate(engagement_score = (num_words + numfaces_project + numfaces_creator) / 3)



all_data$captions <- ifelse(is.na(all_data$captions), 'None', all_data$captions)

# Step 1: Filter rows where captions contain only 1 word
single_word_captions <- all_data %>%
  filter(sapply(strsplit(captions, "\\s+"), length) == 1)

# View the rows with single word captions
print(single_word_captions)

# Step 2: Get unique words from the filtered single word captions
unique_words <- unique(single_word_captions$captions)


# Create a new column in all_data based on the presence of unique words in captions
all_data <- all_data %>%
  mutate(captions_single_word = ifelse(sapply(captions, function(caption) {
    any(unlist(strsplit(caption, "\\s+")) %in% unique_words)
  }), 1, 0))


# Convert 'category_parent' to factor
all_data$category_parent <- as.factor(all_data$category_parent)
all_data$success <- as.factor(all_data$success)

#mode
calculate_mode <- function(x) {
  unique_x <- unique(na.omit(x))  # Remove NA values
  unique_x[which.max(tabulate(match(x, unique_x)))]  # Find the most frequent value
}
mode_value <- calculate_mode(all_data$any_pic)
all_data$any_pic[is.na(all_data$any_pic)] <- mode_value

#all_data <- all_data %>% select(-any pic)

all_data <- all_data %>%
  group_by(creator_id) %>%
  mutate(num_projs = n()) %>%
  ungroup() %>%
  mutate(is_multiple_projs = ifelse(num_projs > 1,1,0))

all_data <- all_data %>%
  group_by(category_parent) %>%
  mutate(category_mean_goal = mean(goal)) %>%
  ungroup() %>%
  mutate(high_for_category = as.factor(ifelse(goal > category_mean_goal, 'YES', 'NO')))



# Calculate the number of projects for each creator
creator_project_count <- all_data %>%
  group_by(creator_id) %>%
  summarise(creator_num_project = n(), .groups = "drop")  # Count the projects for each creator

#creator_project_count <- creator_project_count %>%
# mutate(
#  creator_project_count_1 = ifelse(creator_num_project == 1, 1, 0),
# creator_project_count_2 = ifelse(creator_num_project == 2, 1, 0),
#creator_project_count_2_to_5 = ifelse(creator_num_project > 2 & creator_num_project <= 5, 1, 0),
#creator_project_count_5_plus = ifelse(creator_num_project > 5, 1, 0)
#)

all_data <- all_data %>%
  left_join(creator_project_count, by = "creator_id")

# Add a new column for the length of reward descriptions
all_data <- all_data %>%
  mutate(len_rew_des = nchar(reward_descriptions))

# reward_count
all_data <- all_data %>%
  mutate(reward_count = sapply(reward_amounts, function(x) {
    if (!is.na(x) && x != "") {
      amounts <- as.numeric(unlist(strsplit(as.character(x), ",")))
      return(sum(!is.na(amounts)))
    } else {
      return(0)
    }
  }))


# Step 1: Create a new column by extracting the first value from reward_amounts
all_data <- all_data %>%
  mutate(first_reward_amount = as.numeric(str_extract(reward_amounts, "^[^,]+")))

all_data$first_reward_amount

all_data <- all_data %>%
  mutate(last_reward_amount = sapply(strsplit(reward_amounts, ","), function(x) as.numeric(tail(x, 1))))

all_data <- all_data %>%
  mutate(avg_reward_amount = sapply(strsplit(reward_amounts, ","), function(x) mean(as.numeric(x))))



all_data <- all_data %>%
  mutate(goal_per_day = goal/project_duration)


# Step 2: Left join and create the new feature
all_data <- all_data %>%
  mutate(goal_vs_category_avg = goal / category_mean_goal)

all_data <- all_data %>%
  mutate(avg_reward_amount = sapply(strsplit(reward_amounts, ","), function(x) mean(as.numeric(x))),
         goal_to_avg_reward_ratio = goal / avg_reward_amount)

all_data <- all_data %>%
  mutate(reward_skewness = sapply(strsplit(reward_amounts, ","), function(x) skewness(as.numeric(x))))

#all_data <- all_data %>%
# mutate(tokenized_blurb = strsplit(blurb, "\\s+")) %>%
#rowwise() %>%
#mutate(ttr = length(unique(tokenized_blurb)) / length(tokenized_blurb))







# Create a new column based on the length of reward descriptions
all_data <- all_data %>%
  mutate(reward_length_flag = factor(ifelse(len_rew_des > 50, "YES", "NO")))

# Calculate the success rate for each creator
all_data <- all_data %>%
  mutate(success_numeric = ifelse(success == "YES", 1, ifelse(success == "NO", 0, NA)))

all_data <- all_data %>%
  mutate(launch_quarter = as.factor(quarter(launched_at)),
         is_holiday_season = ifelse(month(launched_at) %in% c(11,12),1,0))



# Weak colors foreground-background flag
all_data$combined_color <- paste(all_data$color_foreground, all_data$color_background, sep = "_")
weak_color_list <- c(
  "Orange_None", "Purple_None", "Purple_Orange", "Red_None", "Teal_None",
  "Teal_Red", "Yellow_None", "Blue_Purple", "Teal_Yellow", "Teal_Grey",
  "Orange_Teal", "Purple_Grey", "Blue_Teal", "Red_Purple", "Red_Teal",
  "Blue_Brown", "Pink_Brown", "Pink_Red", "Teal_Orange", "Teal_Pink",
  "Yellow_Purple", "Orange_Grey", "Pink_White", "Pink_Green", "Brown_Pink",
  "Black_None", "Pink_Blue", "Grey_None", "Orange_Green", "Purple_Red",
  "Purple_Purple", "Red_Black", "Purple_Blue", "Teal_Black", "Green_Yellow",
  "Grey_Green"
)
all_data$weak_color <- ifelse(all_data$combined_color %in% weak_color_list, 1, 0)


# Reward Density feature
calculate_total_reward <- function(reward_string) {
  # Split the string by comma and convert to numeric
  rewards <- as.numeric(unlist(strsplit(reward_string, ",")))
  # Return the sum of rewards
  return(sum(rewards, na.rm = TRUE))
}
all_data <- all_data %>%
  mutate(total_reward = sapply(reward_amounts, calculate_total_reward),  # Replace 'reward_amounts' with the correct column name
         reward_density = total_reward / goal)  # Assuming 'goal' is a numeric column

# Category success rate
tr_dt <- all_data[1:97420, ]  # First 97420 instances are training
ts_dt <- all_data[97421:nrow(all_data), ]  # Last 11308 instances are test
category_success_rates <- tr_dt %>%
  group_by(category_name) %>%
  summarize(category_success_rate = 100 * mean(success_numeric, na.rm = TRUE))
all_data <- all_data %>%
  left_join(category_success_rates, by = "category_name")


all_data <- all_data %>%
  mutate(wordlength_to_goal_ratio = avg_wordlengths / goal)




summary(all_data)


# Apply SMOTE
set.seed(1)  # For reproducibility
smote_result <- SMOTE(success ~ ., data = all_data, perc.over = 100, perc.under = 200)

# Check the result
table(smote_result$success)


#######################################One -Hot Encoding########################
# List of columns to remove (example)
cols_to_remove <- c("id", "creator_id","name", "creator_name", "blurb","deadline", "created_at", "launched_at", "location_slug", "category_name", "accent_color", 
                    "captions", "tag_names", "reward_amounts", "location_type", "reward_descriptions", "success_numeric","reward_length_flag","state_name","launch_weekday", "combined_color","captions_single_word")

# Remove the unimportant columns
removed_columns <- all_data %>%
  select(all_of(cols_to_remove))
all_data_cleaned <- all_data %>%
  select(-all_of(cols_to_remove))


dummy <- dummyVars( success ~ . , data=all_data_cleaned, fullRank = TRUE)
one_hot_projects <- data.frame(predict(dummy, newdata = all_data_cleaned))

one_hot_projects <- cbind(one_hot_projects, removed_columns)
one_hot_projects$success <- as.factor(all_data_cleaned$success)


feats_labeled <- one_hot_projects %>%
  filter(original_settr == 1) %>%  
  select(-original_settr)

feats_unlabeled <- one_hot_projects %>%
  filter(original_settr == 0) %>%  
  select(-original_settr) 


feats_labeled$success <- as.factor(feats_labeled$success)

#######################################F-Selector###############################

library(FSelector)
set.seed(1)
va_inds <- sample(nrow(feats_labeled), 0.3 * nrow(feats_labeled))
feats_train <- feats_labeled[-va_inds, ]
feats_valid <- feats_labeled[va_inds, ]
weights <- information.gain(success ~ ., feats_train)
top_features <- cutoff.k(weights, 50)
print(top_features)





############################### K fold cross validation#########################
#split train and test data
training_data <- filter(one_hot_projects, original_settr == 1) %>%
  select( -original_settr)  # Remove 'original_set'


testing_data <- filter(one_hot_projects, original_settr == 0) %>%
  select(-original_settr)  # Remove 'original_set'


labeled_shuffle <- training_data[sample(nrow(training_data)),]
# Define k = the number of folds
k <- 10

# Separate data into k equally-sized folds
folds <- cut(seq(1, nrow(labeled_shuffle)), breaks=k, labels=FALSE)

# Initialize performance vectors
log1_accs = rep(0, k)
log2_accs = rep(0, k)
log3_accs = rep(0, k)
log4_accs = rep(0, k)
log5_accs = rep(0, k)


log2formula = success ~ num_words + avg_wordlengths + afinn_pos + afinn_neg +
  NOUN + ADP + DET + PRON + VERB + NUM + CONJ + ADJ + log_goal + grade_level  + len_rew_des
formula4 = success ~ reward_count + log_goal +
  interaction_goal_avgwordlength + category_mean_goal + len_rew_des +
  NOUN + ADJ + ADP + num_words + NUM +
  project_duration + sentence_counter + DET + afinn_pos +
  high_for_category.YES + CONJ + VERB + PRON + 
  afinn_neg + ADV + avg_wordlengths + category_parent.music +
  PRT + category_parent.theater + num_projs + 
  more_avg_sentence + verbs_per_sentence + state_nameny +
  avgsyls + interaction_goal_numwords

log1formula <- success ~ reward_count + goal + interaction_goal_avgwordlength + category_mean_goal + len_rew_des + NOUN + ADJ + ADP + num_words + NUM + project_duration + sentence_counter + DET + afinn_pos + high_for_category.YES + CONJ + VERB + PRON + afinn_neg + ADV + avg_wordlengths + category_parent.music + PRT + category_parent.theater + num_projs + more_avg_sentence + verbs_per_sentence + state_nameny + avgsyls + interaction_goal_numwords + category_parent.dance + state_namefl + regionMidAtl + is_multiple_projs + regionSouthAtl + time_to_launch + avgsentencelength + words_per_sentence + grade_level + tag_popularity + category_parent.fashion + location_typeTown + is_Town.1

log5formula <- success ~ project_duration + goal + tag_count + verbs_per_sentence + male_dominant_project + is_Town.1 + any_pic + is_sentiment_positive + blurb_sentiment + tag_popularity + is_multiple_projs + high_for_category.YES + PRON + len_rew_des + NOUN + ADJ + ADP + num_words + NUM

for(i in 1:k) {
  valid_inds <- which(folds == i, arr.ind = TRUE)
  valid_fold <- labeled_shuffle[valid_inds, ]
  train_fold <- labeled_shuffle[-valid_inds, ]
  
  
  valid_actuals <- labeled_shuffle$success[valid_inds]
  
  # Train model and predict
  tr_pred <- function(train_data, valid_data, model_formula){
    trained_tree <- rpart(model_formula, data = train_data, cp = -1, minbucket = 2, maxdepth = 7)
    predictions <- predict(trained_tree, newdata = valid_data, type = "prob")[,2]
    return(predictions)
  }
  
  classify <- function(scores, c){
    classifications <- ifelse(scores > c, "YES", "NO")
    return(classifications)
  }
  
  
  # Tree Model 1
  probs1 <- tr_pred(train_fold, valid_fold, log1formula)
  classifications1 <- classify(probs1, 0.55)
  valid_actuals <- factor(valid_actuals, levels = c("YES", "NO"))
  class1 <- factor(classifications1, levels = levels(valid_actuals))
  CM_log1 <- confusionMatrix(data = class1, reference = valid_actuals, positive = "YES")
  accuracy <- as.numeric(CM_log1$overall["Accuracy"])
  log1_accs[i] <- accuracy
  
  #Tree Model 2
  probs2 <- tr_pred(train_fold, valid_fold, log2formula)
  classifications2 <- classify(probs2, 0.55)
  valid_actuals2 <- factor(valid_actuals, levels = c("YES", "NO"))
  class2 <- factor(classifications2, levels = levels(valid_actuals))
  CM_log2 <- confusionMatrix(data = class2, reference = valid_actuals, positive = "YES")
  accuracy2 <- as.numeric(CM_log2$overall["Accuracy"])
  log2_accs[i] <- accuracy2
  
  #Tree Model 3
  probs4 <- tr_pred(train_fold, valid_fold, formula4)
  classifications4 <- classify(probs4, 0.55)
  valid_actuals4 <- factor(valid_actuals, levels = c("YES", "NO"))
  class4 <- factor(classifications4, levels = levels(valid_actuals))
  CM_log4 <- confusionMatrix(data = class4, reference = valid_actuals, positive = "YES")
  accuracy4 <- as.numeric(CM_log4$overall["Accuracy"])
  log4_accs[i] <- accuracy4
  
  #Tree Model 3
  probs5 <- tr_pred(train_fold, valid_fold, log5formula)
  classifications5 <- classify(probs5, 0.55)
  valid_actuals5 <- factor(valid_actuals, levels = c("YES", "NO"))
  class5 <- factor(classifications5, levels = levels(valid_actuals))
  CM_log5 <- confusionMatrix(data = class5, reference = valid_actuals, positive = "YES")
  accuracy5 <- as.numeric(CM_log5$overall["Accuracy"])
  log5_accs[i] <- accuracy5
}

#######################################PLOTTING#################################
# Set up the plot with the correct limits
plot(c(1:k), log1_accs, type = 'l', col = 'red', 
     ylim = range(c(log1_accs, log2_accs, log5_accs, log4_accs), na.rm = TRUE), 
     xlab = "Fold", ylab = "Accuracy", main = "Model Performance Across Folds")

# Add points for model 1 (log1)
points(c(1:k), log1_accs, col = 'red')

# Add lines and points for model 2 (log2)
lines(c(1:k), log2_accs, type = 'l', col = 'blue')
points(c(1:k), log2_accs, col = 'blue')

# Add lines and points for model 3 (log3)
lines(c(1:k), log5_accs, type = 'l', col = 'green')
points(c(1:k), log5_accs, col = 'green')

# Add lines and points for model 4 (tree4)
lines(c(1:k), log4_accs, type = 'l', col = "purple")
points(c(1:k), log4_accs, col = 'purple')

# Add a legend to differentiate the models
legend("bottomright", legend = c("Model 1 (log1)", "Model 2 (log2)", "Model 3 (log3)", "Model 4 (log4)"), 
       col = c("red", "blue", "green", "purple"), lty = 1, pch = 1)

print(log4_accs)



##### XG Boost

library(ranger)
library(xgboost)
library(vip)
library(pROC)  # For AUC calculation
# Convert the 'success' variable to numeric (1 for 'YES', 0 for 'NO')


one_hot_projects$success_numeric <- ifelse(one_hot_projects$success == "YES", 1, 0)

# Remove any problematic or unnecessary columns from the data before proceeding
# Assuming you want to drop any previously problematic columns (you can adjust as needed)
cols_to_remove <- c("id", "creator_id", "name", "creator_name", "blurb", "deadline", "created_at", 
                    "launched_at", "location_slug", "category_name", "accent_color", "captions", 
                    "tag_names", "reward_amounts", "reward_descriptions", "success","interaction_goal_numwords","reward_length_flag","state_name", "location_type", "agediff_project", "is_Town.1", "ADP", "PRT", "DET", "CONJ", "combined_color","captions_single_word","launch_weekday")

# Cleaned dataset
one_hot_projects_cleaned <- one_hot_projects %>%
  select(-all_of(cols_to_remove))  # Remove the unwanted columns


training_data <- filter(one_hot_projects_cleaned, original_settr == 1) %>%
  select( -original_settr)  # Remove 'original_set'

#summary(training_data)

testing_data <- filter(one_hot_projects_cleaned, original_settr == 0) %>%
  select(-original_settr)  # Remove 'original_set'

colnames(training_data)
# Split the dataset into training and validation (70-30 split)
set.seed(1)  # Ensure reproducibility
train_indices <- createDataPartition(training_data$success_numeric, p = 0.70, list = FALSE)

# Create training and validation sets
x_train <- as.matrix(training_data[train_indices, ] %>% select(-success_numeric))  # Features for training
y_train_num <- training_data$success_numeric[train_indices]  # Target variable for training

x_valid <- as.matrix(training_data[-train_indices, ] %>% select(-success_numeric))  # Features for validation
y_valid_num <- training_data$success_numeric[-train_indices]  # Target variable for validation

#one_hot_projects_cleaned <- one_hot_projects_cleaned %>%
# select_if(is.numeric)

colnames(one_hot_projects_cleaned)

########################################
# Train the initial XGBoost model
########################################
boost.mod <- xgboost(
  data = x,
  label = y_train_numeric,
  max.depth = 5,
  nrounds = 500,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  lambda = 4,
  alpha = 0.5,
  objective = "binary:logistic",
  verbosity = 0
)

vip(boost.mod,num_features = 20)

# Make predictions on validation set
boost_preds <- predict(boost.mod, x_valid)

# Calculate AUC for validation set
auc_valid <- roc(y_valid_num, boost_preds)$auc
print(paste("Initial Model AUC on Validation Set:", auc_valid))


# Calculate accuracy for validation set
boost_classifications <- ifelse(boost_preds > 0.53, 1, 0)
accuracy_valid <- mean(boost_classifications == y_valid_num)
print(paste("Initial Model Accuracy on Validation Set:", accuracy_valid))


boost_preds <- predict(boost.mod, x_test)
classifications_success <- ifelse(boost_preds > 0.53, "YES", "NO")


# Save or output the classifications_success
classifications_success <- factor(classifications_success, levels = c("YES", "NO"))

#this code creates a sample output in the correct format
write.table(classifications_success, "success_group4_FINAL.csv", row.names = FALSE)


##############################################
# Hyperparameter Grid Search Function
##############################################
grid_search <- function() {
  
  # Hyperparameter values to search over
  depth_choose <- c(5,7,10)
  nrounds_choose <- c(500,1000,1500,2000)
  eta_choose <- c(0.01, 0.05, 0.1,0.3)
  
  grid_size <- length(depth_choose) * length(nrounds_choose) * length(eta_choose)
  grid_store <- matrix(nrow = grid_size, ncol = 6)  # Add column for accuracy
  
  inner_counter <- 0
  
  # Nested loops for grid search over three hyperparameters
  print('depth, nrounds, eta, AUC, Accuracy')
  for (i in 1:length(depth_choose)) {
    for (j in 1:length(nrounds_choose)) {
      for (k in 1:length(eta_choose)) {
        
        this_depth <- depth_choose[i]
        this_nrounds <- nrounds_choose[j]
        this_eta <- eta_choose[k]
        
        # Train model for current combination of hyperparameters
        inner_bst <- xgboost(
          data = x_train, 
          label = y_train_num, 
          max.depth = this_depth, 
          eta = this_eta, 
          nrounds = this_nrounds,
          objective = "binary:logistic", 
          verbosity = 0,
          verbose = 0
        )
        
        # Make predictions on validation set
        inner_bst_pred <- predict(inner_bst, x_valid)
        
        # Calculate AUC on validation set
        inner_bst_auc <- roc(y_valid_num, inner_bst_pred)$auc
        
        # Calculate accuracy for validation set
        inner_bst_classifications53 <- ifelse(inner_bst_pred > 0.53, 1, 0)
        inner_bst_accuracy53 <- mean(inner_bst_classifications53 == y_valid_num)
        
        inner_bst_classifications55 <- ifelse(inner_bst_pred > 0.55, 1, 0)
        inner_bst_accuracy55 <- mean(inner_bst_classifications55 == y_valid_num)
        
        
        # Store performance
        inner_counter <- inner_counter + 1
        grid_store[inner_counter, 1] <- this_depth
        grid_store[inner_counter, 2] <- this_nrounds
        grid_store[inner_counter, 3] <- this_eta
        grid_store[inner_counter, 4] <- inner_bst_auc
        grid_store[inner_counter, 5] <- inner_bst_accuracy53
        grid_store[inner_counter, 6] <- inner_bst_accuracy55
        
        # Print the performance for each hyperparameter combination
        print(paste(inner_counter, this_depth, this_nrounds, this_eta, inner_bst_auc, inner_bst_accuracy53, inner_bst_accuracy55, sep = ", "))
      }
    }
  }
  
  return(grid_store)
}

# Uncomment the line below to run the hyperparameter grid search
grid_search_results <- grid_search()

x <- as.matrix(training_data %>% select(-success_numeric))  # Features for training
y_train_numeric <- training_data$success_numeric  # Target variable for training
summary(x)

x_test <- as.matrix(testing_data %>% select(-success_numeric))

inner_bst <- xgboost(
  data = x_train, 
  label = y_train_num, 
  max.depth = 7, 
  eta = 0.2, 
  nrounds = 650,  
  objective = "binary:logistic", 
  verbosity = 0  # Silence output
)


# Make predictions on validation set
inner_bst_pred <- predict(inner_bst, x_valid)

# Calculate accuracy for validation set
inner_bst_classifications <- ifelse(inner_bst_pred > 0.53, 1, 0)

inner_bst_accuracy <- mean(inner_bst_classifications == y_valid_num)
inner_bst_accuracy
classifications_success <- ifelse(inner_bst_pred > 0.53, "YES", "NO")


# Save or output the classifications_success
classifications_success <- factor(classifications_success, levels = c("YES", "NO"))

#this code creates a sample output in the correct format
write.table(classifications_success, "success_group4_C7.csv", row.names = FALSE)

# Find the best hyperparameter combination based on AUC
best_hp_row <- which.max(grid_search_results[, 5])  # Index of the best AUC score
best_depth <- grid_search_results[best_hp_row, 1]
best_nrounds <- grid_search_results[best_hp_row, 2]
best_eta <- grid_search_results[best_hp_row, 3]

# Train the best model on the remaining training data with the best hyperparameters
best_bst <- xgboost(
  data = x_train, 
  label = y_train_num, 
  max.depth = best_depth, 
  eta = best_eta, 
  nrounds = best_nrounds,  
  objective = "binary:logistic", 
  verbosity = 0  # Silence output
)

# Make predictions on validation set with best model
best_bst_pred <- predict(best_bst, x_valid)

# Calculate final AUC and accuracy on validation set
final_auc <- roc(y_valid_num, best_bst_pred)$auc
final_classifications <- ifelse(best_bst_pred > 0.53, 1, 0)
final_accuracy <- mean(final_classifications == y_valid_num)

# Output final AUC and accuracy
print(paste("Best Model AUC on Validation Set:", final_auc))
print(paste("Best Model Accuracy on Validation Set:", final_accuracy))

# Generate confusion matrix for validation set
conf_matrix <- confusionMatrix(
  factor(final_classifications, levels = c(0, 1)), 
  factor(y_valid_num, levels = c(0, 1)), 
  positive = "1"
)
print(conf_matrix)

#######################################TEST DATA###############################

log1formula <- success ~ reward_count + goal + interaction_goal_avgwordlength + category_mean_goal + len_rew_des + NOUN + ADJ + ADP + num_words + NUM + project_duration + sentence_counter + DET + afinn_pos + high_for_category.YES + CONJ + VERB + PRON + afinn_neg + ADV + avg_wordlengths + category_parent.music + PRT + category_parent.theater + num_projs + more_avg_sentence + verbs_per_sentence + state_nameny + avgsyls + interaction_goal_numwords + category_parent.dance + state_namefl + regionMidAtl +  is_multiple_projs + regionSouthAtl + time_to_launch + avgsentencelength + words_per_sentence + grade_level + tag_popularity + category_parent.fashion + location_typeTown + is_Town.1

final_model <- rpart(log1formula, data = training_data, cp = -1, minbucket = 2, maxdepth = 7)

# Predict success for test_x using the final model
test_probs <- predict(final_model, newdata = testing_data, type = "prob")[, 2]

# Classify the predictions based on a cutoff (0.6)
cutoff <- 0.55
classifications_success <- ifelse(test_probs > cutoff, "YES", "NO")

# Save or output the classifications_success
classifications_success <- factor(classifications_success, levels = c("YES", "NO"))

#this code creates a sample output in the correct format
write.table(classifications_success, "success_group4_check.csv", row.names = FALSE)