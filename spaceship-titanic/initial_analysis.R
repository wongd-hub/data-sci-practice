## Load packages ----
invisible({

  # Packages to load are found in the character vector below
  packages <- c(
    "tidyverse", "data.table", "ggplot2", 
    "scales", "tictoc", "infotheo", 
    "Information", "gridExtra", "mice"
  )

  if(!all(packages %in% rownames(installed.packages()))) {
  
    to_install <- packages[!(packages %in% rownames(installed.packages()))]
    lapply(to_install, install.packages, character.only = TRUE)
    rm(to_install)

  }

  lapply(packages, library, character.only = TRUE)
  rm(packages)

})

setwd('~/Documents/Projects/kaggle/spaceship-titanic')

## Introduction ----
# This is a solution to the Kaggle: Spaceship Titanic competition https://www.kaggle.com/competitions/spaceship-titanic/overview
# The issue is a binary classification problem, determining whether a given individual has been transported to another dimension

## Load data & manipulate ----
train <- fread('raw-data/train.csv') %>% as_tibble()
test  <- fread('raw-data/test.csv') %>% as_tibble()

all   <- bind_rows(
  train %>% mutate(dataset = 'train'),
  test %>% mutate(dataset = 'test')
) 

## 1a. EDA - Categorical Variables ----
# Getting a glimpse of the data
#  We have 15 variables and 12,970 observations
glimpse(all)

# View cardinality of variables
all %>% 
  select(-dataset) %>% 
  sapply(n_distinct)

# See distinct values of all present categorical variables
#  Potentially interesting for feature engineering:
#   Passenger ID is in form gggg_pp, with gggg representing the travel group and pp representing passenger within group
#   Cabin is in the form deck/num/side. Side can be P (Port) or S (Starboard).
#   Name can be split into first and last name, although using last name to derive family may not be much better than using group 
#    from Passenger ID
all %>% 
  select(-dataset) %>% 
  select_if(is.character) %>% 
  sapply(unique)

# Charts to see distributions of low cardinality categorical variables and booleans
#  Wary of VIP as only a small percentage are VIPs - even smaller than percentage of NAs
all %>% 
  # Variables of interest
  select(HomePlanet, Destination, CryoSleep, VIP) %>% 
  gather(variable, value) %>% 
  # Calculating percentage distributions for each value
  group_by(variable, value) %>% 
  summarise(
    n = n(),
    .groups = 'drop'
  ) %>% 
  group_by(variable) %>% 
  mutate(percentage = n / sum(n)) %>% 
  ungroup() %>% 
  # Charting
  ggplot(aes(x = value, y = percentage, label = percent(percentage, accuracy = 0.1))) +
  geom_col() +
  facet_wrap(vars(variable), scales = 'free') +
  #  Label formatting
  labs(x = 'Value', y = 'Percent') +
  geom_text(
    position = position_dodge(width = .9),
    vjust = -0.5,
    size = 3
  ) + 
  scale_y_continuous(labels = percent)

# See how these 4 variables vary with the target value - these tables have row-wise percentages
#  People from Europa seem to be transported at a higher frequency than people from other planets
#  People from Earth seem to be transported at a lower frequency
table(train$HomePlanet, train$Transported) %>% prop.table(1)

#  People headed to 55 Cancri e seem to be transported at a higher frequency than the other destinations
table(train$Destination, train$Transported) %>% prop.table(1)

#  People in CryoSleep seem to be transported at a higher frequency than not
table(train$CryoSleep, train$Transported) %>% prop.table(1)

#  People in VIP tend to be transported less - however be wary of this variable as there is only a small pct of VIPs
table(train$VIP, train$Transported) %>% prop.table(1)

# Deeper dive into Cabins
#  Split into Deck/Num/Side - we'll discuss Num in the numeric variables section
all_cabin_spl <- all %>% 
  mutate(Cabin = na_if(Cabin, "")) %>% 
  separate(Cabin, c("Deck", "Num", "Side"), "/")

#  View cardinality
all_cabin_spl %>% 
  select(Deck, Num, Side) %>% 
  sapply(n_distinct)

#  Distinct values for categorical variables and distribution chart
all_cabin_spl %>% 
  select(Deck, Side) %>% 
  sapply(unique)

#   Passengers tend to be heavily weighted towards Decks F and G
all_cabin_spl %>% 
  # Variables of interest
  select(Deck, Side) %>% 
  gather(variable, value) %>% 
  # Calculating percentage distributions for each value
  group_by(variable, value) %>% 
  summarise(
    n = n(),
    .groups = 'drop'
  ) %>% 
  group_by(variable) %>% 
  mutate(percentage = n / sum(n)) %>% 
  ungroup() %>% 
  # Charting
  ggplot(aes(x = value, y = percentage, label = percent(percentage, accuracy = 0.1))) +
  geom_col() +
  facet_wrap(vars(variable), scales = 'free') +
  #  Label formatting
  labs(x = 'Value', y = 'Percent') +
  geom_text(
    position = position_dodge(width = .9),
    vjust = -0.5,
    size = 3
  ) + 
  scale_y_continuous(labels = percent)

#  See how deck and side correspond to being transported
train_cabin_spl <- all_cabin_spl %>% filter(dataset == 'train') %>% select(-dataset)

#   Decks B, C, E, T all seem to be transported at a varying rate - note though that T is a very small deck (0.1% of all passengers)
table(train_cabin_spl$Deck, train_cabin_spl$Transported) %>% prop.table(1)

#   Doesn't look like Side matters too much here
table(train_cabin_spl$Side, train_cabin_spl$Transported) %>% prop.table(1)

# Deeper dive into passenger ID and travel groups
#  Let's create a variable to see the size of an individual's travelling group
all_grp_sep <- all %>% 
  separate(PassengerId, c('GroupId', 'PassengerId'), "_")

all_grps <- all_grp_sep %>% 
  distinct(GroupId, PassengerId) %>% 
  group_by(GroupId) %>% 
  summarise(
    group_size = n(), 
    .groups = 'drop'
  )

#  View distribution of group sizes
all_grps %>% 
  ggplot(aes(x = group_size)) +
  geom_histogram() +
  scale_x_continuous(breaks = unique(all_grps$group_size) %>% sort()) +
  scale_y_continuous(labels = comma) +
  labs(x = 'Group Size', y = "Count", title = "Distribution of Travel Group Sizes")

all_grouped <- all_grp_sep %>% 
  left_join(all_grps, by = 'GroupId')

train_grouped <- all_grouped %>% filter(dataset == 'train') %>% select(-dataset)

#  Looks like transportation rate varies a little based on travelling group size
table(train_grouped$group_size, train_grouped$Transported) %>% prop.table(1)

#  Determining whether the passenger is female or male using first name might be something for further review

## 1b. EDA - Numeric Variables ----
# The following numeric variables are available to us
all %>% 
  select(-dataset) %>% 
  select_if(is.numeric) %>% 
  summary()

#  As well as the room number of an individual
all_cabin_spl %>% 
  select(Num) %>% 
  mutate(Num = as.numeric(Num)) %>% 
  summary()

# View histograms to see if distribution changes by transportation status, indicating possibly useful features
#  Nothing seems to jump out as particularly illuminating, although it seems like younger children are more likely to be transported
train_cabin_spl %>% 
  mutate(Num = as.numeric(Num)) %>% 
  select_if(sapply(., is.numeric) | str_detect(names(.), "Transported")) %>% 
  gather(variable, value, -Transported) %>% 
  ggplot(aes(x = value, fill = Transported)) +
  geom_histogram(position = 'dodge') +
  facet_wrap(vars(variable), scales = 'free') +
  scale_y_continuous(labels = comma) +
  theme(legend.position = "bottom") + 
  labs(x = 'Value', y = 'Count')

## 2. Feature Importance ----
# While we've developed hypotheses regarding which features may have value in predicting whether an individual has been transported
#  or not, let's back these with some formal analysis.

all_wrk <- all_cabin_spl %>% 
  separate(PassengerId, c('GroupId', 'PassengerId'), "_") %>% 
  left_join(all_grps, by = 'GroupId') %>% 
  mutate_all(.funs = ~ na_if(., ""))

train_wrk <- all_wrk %>% filter(dataset == 'train') %>% select(-dataset)

# First we'll use mutual information to rank the variables in terms of information level
# train_wrk %>% 
#   select(-Transported) %>% 
#   lapply(function(x) mutinformation(., train_wrk$Transported))

mutual_info <- tibble(
  variable = train_wrk %>% select(-Transported, -GroupId, -Name, -PassengerId, -Num) %>% colnames(),
  mutual_info = NA_real_
)

for (i in 1:nrow(mutual_info)) {

  mutual_info[[i, 2]] <- mutinformation(train_wrk[mutual_info[[i, 1]]], train_wrk$Transported)
  
}

mutual_info_fct_order <- mutual_info %>% arrange(mutual_info) %>% pull(variable)

mutual_info <- mutual_info %>% 
  mutate(variable = factor(variable, levels = mutual_info_fct_order))

mutual_info %>% 
  ggplot(aes(x = mutual_info, y = variable)) +
  geom_col() +
  labs(x = 'Mutual Information', y = 'Variable')

# Note how the following potential candidates for features have low Mutual Information: HomePlanet, Destinatino, Deck, Age


# Next, we'll look at Information Value/Weight of Evidence which is related but slightly different: 
#  https://stats.stackexchange.com/questions/16945/why-do-people-use-the-term-weight-of-evidence-and-how-does-it-differ-from-poi
train_woe <- train_wrk %>% 
  select(-GroupId, -Name, -PassengerId) %>% 
  # Convert all character columns to factors
  mutate_if(
    is.character,
    factor
  ) %>% 
  # Force Transported to numeric
  mutate(Transported = as.numeric(Transported))

# Information values > 0.1 have medium strength
inf_value <- create_infotables(data = train_woe, y = "Transported", bins = 10)$Summary

# Comparing Mutual Information and Information Value
#  Generally the two measures agree in terms of ranking feature strength.
#  Information Values > 0.1 indicate a medium strength predictive power, so we'll start with these variables
grid.arrange(
  mutual_info %>% 
    ggplot(aes(x = mutual_info, y = variable)) +
    geom_col() +
    labs(x = 'Mutual Information', y = 'Variable'),
  inf_value %>% 
    mutate(Variable = factor(Variable, levels = inf_value %>% arrange(IV) %>% pull(Variable))) %>% 
    ggplot(aes(x = IV, y = Variable)) +
    geom_col() +
    labs(x = 'Information Value', y = element_blank()) +
    geom_vline(xintercept = 0.1),
  ncol = 2
)

## 3. Missing/NAs ----
# Noting some missing values hidden as empty strings, convert to NAs and count number of missings
#  In total, 6,364 / 12,970rows have at least one NA in them; definitely don't want to cut 50% of observations out of the dataset
all_wrk[rowSums(is.na(all_wrk %>% mutate_all(.funs = ~ na_if(., "")))) > 0, ] %>% nrow()

# Where do these missing values reside?
(all_wrk %>% 
  mutate_all(.funs = ~ na_if(., "")) %>% 
  sapply(function(x) sum(is.na(x))) / nrow(all_wrk)) %>% 
  as.list() %>% 
  data.frame() %>% 
  gather(variable, missing_pct) %>% 
  arrange(desc(missing_pct))

# Missing value handling
#  Name will not be used as a feature so we can leave this for now. Same as Age, VIP, and Destination.
#  The numeric values such as RoomService, FoodCourt, ShoppingMall, Spa, VRDeck will be predicted using mice
#  Cabin contains Deck which will need to be imputed - first determine if they're in a group and assign them to that room (assuming shared rooms)
#  HomePlanet - will see if this is related to Cabin at all
#  CryoSleep are confined to their quarters so will have likely spent no money on other amenities

# Imputation
#  Luxury Amenities part 1 - CryoSleeping Individuals
#   Individuals in CryoSleep are confined to their Cabin throughout the duration of the interstellar trip,
#   this means that they are likely not to have spent any money on the additional luxury amenities onboard the ship.
all_cryo_sub <- all_wrk %>% 
  mutate(lux_spend = ShoppingMall + VRDeck + FoodCourt + Spa + RoomService)

all_cryo_sub %>% filter(CryoSleep & lux_spend > 0) %>% nrow() == 0 # No-one in CryoSleep has spent on amenities

# We can then infer that all luxury amenities will be 0 for people in CryoSleep - this will fix luxury amenities for 513 observations
all_cryo_sub %>% filter(CryoSleep & is.na(lux_spend)) %>% nrow()

# We now replace all NAs in the luxury amenities with 0 where CryoSleep is TRUE
all_wrk <- all_wrk %>% 
  mutate_at(
    .vars = vars(ShoppingMall, VRDeck, FoodCourt, Spa, RoomService),
    .funs = ~ ifelse(CryoSleep & is.na(.), 0, .)
  )


#  Luxury Amenities part 2 - Non-CryoSleeping Individuals

# Viewing relationship to other variables
# all_wrk %>% 
#   filter(!CryoSleep) %>% 
#   mutate(lux_spend = ShoppingMall + VRDeck + FoodCourt + Spa + RoomService) %>% 
#   ggplot(aes(y = lux_spend, x = Deck, fill = Side)) +
#   geom_boxplot() +
#   scale_y_continuous(labels = dollar)

# all_wrk %>% 
#   filter(!CryoSleep) %>% 
#   mutate(lux_spend = ShoppingMall + VRDeck + FoodCourt + Spa + RoomService) %>% 
#   ggplot(aes(y = lux_spend, x = Destination, fill = HomePlanet)) +
#   geom_boxplot() +
#   scale_y_continuous(labels = dollar)

# all_wrk %>%
#   filter(!CryoSleep) %>%
#   mutate(lux_spend = ShoppingMall + VRDeck + FoodCourt + Spa + RoomService) %>%
#   ggplot(aes(y = lux_spend, x = as.factor(group_size))) +
#   geom_boxplot() +
#   scale_y_continuous(labels = dollar)

all_wrk %>%
  filter(!CryoSleep) %>%
  mutate(lux_spend = ShoppingMall + VRDeck + FoodCourt + Spa + RoomService) %>%
  mutate(Num = as.numeric(Num)) %>%
  ggplot(aes(x = Num, y = lux_spend, colour = HomePlanet)) +
  geom_point() +
  scale_y_continuous(labels = dollar) +
  scale_x_continuous(labels = comma)
#  Side note, HomePlanet appears to be related to room number a bit - Europa are all low ~<500 numbers


#   Amenities spend appears to be related to Destination, HomePlanet, Deck, attempt imputation via 
#    linear regression
all_wrk_noncryo <- all_wrk %>% filter(!CryoSleep)
  
imput_lux_spend <- all_wrk_noncryo %>% 
  select(HomePlanet, Destination, Deck, ShoppingMall, VRDeck, FoodCourt, Spa, RoomService) %>% 
  mice(method = 'rf') %>% 
  complete()

#   Compare imputed values vs. non-imputed
bind_rows(
  all_wrk_noncryo %>% 
    select(ShoppingMall, VRDeck, FoodCourt, Spa, RoomService) %>% 
    gather(var, value) %>% 
    mutate(dataset = 'Non-Imputed'),
  imput_lux_spend %>% 
    select(ShoppingMall, VRDeck, FoodCourt, Spa, RoomService) %>% 
    gather(var, value) %>% 
    mutate(dataset = 'Imputed')
) %>% 
  ggplot(aes(x = value)) +
  geom_histogram() +
  facet_grid(rows = vars(var), cols = vars(dataset), scales = 'free') +
  scale_x_continuous(labels = dollar) +
  scale_y_continuous(labels = comma) +
  labs(x = 'Spend', y = 'Count')

#  Add these values back to the dataset and dive deeper (label rows with NAs with 'imputed' flag)

## 4. Training ----
# Split train dataset into training and validation sets
train_set <- sample(nrow(GermanCredit), round(0.6*nrow(GermanCredit)))