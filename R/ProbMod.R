install.packages("data.table")


library(data.table)
library(dplyr)


data <- as.data.frame(fread("/Users/Moritz/Desktop/dataset/data.tsv"))
data2 <- as.data.frame(fread("/Users/Moritz/Desktop/dataset/data2.tsv"))
data3 <- as.data.frame(fread("/Users/Moritz/Desktop/dataset/data3.tsv"))
data4 <- as.data.frame(fread("/Users/Moritz/Desktop/dataset/data4.tsv"))
data5 <- as.data.frame(fread("/Users/Moritz/Desktop/dataset/data5.tsv"))
data6 <- as.data.frame(fread("/Users/Moritz/Desktop/dataset/data6.tsv"))

#cleansing
data <- select(data, -c(job, characters))
data3 <- select(data3, -c(writers))
data4 <- select(data4, -c(titleType, isAdult, endYear ))
data5 <- select(data5, -c(ordering:title, language, attributes, isOriginalTitle))
data6 <- select(data6, -c(deathYear:knownForTitles))

#merging
adata <- merge(data6, data, by = "nconst", all = FALSE, sort = FALSE)
mdata <- merge(data2, data4, by = "tconst", all = FALSE, sort = FALSE)

colnames(data5)[1] <- "tconst"

mdata <- merge(mdata, data5, by = "tconst", all = FALSE, sort = FALSE)

data4_2 <- select(data4, -c(primaryTitle, originalTitle, runtimeMinutes, genres ))

load("adata.Rda")
load("mdata.Rda")
load("data4_2.Rda")


adata <- merge(adata, data4_2, by = "tconst", all = FALSE, sort = FALSE)

adata <- adata %>% filter(adata$startYear >= 2015)
mdata <- mdata %>% filter(mdata$startYear >= 2015)


fdata <- merge(mdata, adata, by ="tconst", all = FALSE, sort = FALSE)


# the-numbers data set
library(readxl)

numbers <- read_excel("/Users/Moritz/Desktop/dataset/numbers1.xlsx")

colnames(fdata)[4] <- "Title"

fdata_2 <- merge(numbers, fdata, by ="Title", all = FALSE, sort = FALSE)

length(unique(fdata_2$Title))

#cleansing

unique(fdata_2$category)

fdata_2 <- select(fdata_2, -c(tconst, originalTitle, region:nconst))

fdata_2$`Production Budget`  <- gsub('[$]','',fdata_2$`Production Budget`)
fdata_2$`Production Budget` <- as.numeric(gsub("," ,"", fdata_2$`Production Budget`))

fdata_2$`International Box Office`  <- gsub('[$]','',fdata_2$`International Box Office`)
fdata_2$`International Box Office` <- as.numeric(gsub("," ,"", fdata_2$`International Box Office`))

fdata_2$`Opening Weekend Revenue`  <- gsub('[$]','',fdata_2$`Opening Weekend Revenue`)
fdata_2$`Opening Weekend Revenue` <- as.numeric(gsub("," ,"", fdata_2$`Opening Weekend Revenue`))

fdata_2$`Domestic Box Office`  <- gsub('[$]','',fdata_2$`Domestic Box Office`)
fdata_2$`Domestic Box Office` <- as.numeric(gsub("," ,"", fdata_2$`Domestic Box Office`))

fdata_2$`Worldwide Box Office`  <- gsub('[$]','',fdata_2$`Worldwide Box Office`)
fdata_2$`Worldwide Box Office` <- as.numeric(gsub("," ,"", fdata_2$`Worldwide Box Office`))

fdata_2 <- fdata_2[,-2] # delet irrelevant variable

#new variables

colnames(fdata_2)[12] <- "SCREENS" #opening weekend theatres

CUMVIEWERS <- as.data.frame(fdata_2$`Opening Weekend Revenue`/fdata_2$SCREENS) #variable CUMVIEWERS
fdata_2["CUMVIEWERS"] <- CUMVIEWERS

unique(fdata_2$`Theatrical Distributor`)
fdata_2$DIST <- ifelse(fdata_2$`Theatrical Distributor`%in% c("Walt Disney","Sony Pictures",
                                                              "Warner Bros.","20th Century Fox",
                                                              "Paramount Pictures","Universal"),1,0) # 1 if top 6 otherwise 0

#Genre variables
unique(fdata_2$Genre)

fdata_2$adventure <- ifelse(fdata_2$Genre %in% c("Adventure"), 1, 0)
fdata_2$blcomedy <- ifelse(fdata_2$Genre %in% c("Black Comedy"), 1, 0)
fdata_2$comedy <- ifelse(fdata_2$Genre %in% c("Comedy"), 1, 0)
fdata_2$docu <- ifelse(fdata_2$Genre %in% c("Documentary"), 1, 0)
fdata_2$drama <- ifelse(fdata_2$Genre %in% c("Drama"), 1, 0)
fdata_2$horror <- ifelse(fdata_2$Genre %in% c("Horror"), 1, 0)
fdata_2$musical <- ifelse(fdata_2$Genre %in% c("Musical"), 1, 0)
fdata_2$romcomedy <- ifelse(fdata_2$Genre %in% c("Romantic Comedy"), 1, 0)
fdata_2$thriller <- ifelse(fdata_2$Genre %in% c("Thriller/Suspense"), 1, 0)
fdata_2$western <- ifelse(fdata_2$Genre %in% c("Western"), 1, 0)
fdata_2$action <- ifelse(fdata_2$Genre %in% c("Action"), 1, 0)

####### director dataset

library(plyr)
dat <- read.csv('/Users/Moritz/Desktop/dataset/movie_metadata.csv', header=TRUE)
df <- as.data.frame(dat)
head(df)

count(unique(df$actor_1_name))

# calculate the mean rating and SE for each main actor
ratingdat <- ddply(df, c("actor_1_name"), summarise,
                   M = mean(imdb_score, na.rm=T),
                   SE = sd(imdb_score, na.rm=T)/sqrt(length(na.omit(imdb_score))),
                   N = length(na.omit(imdb_score)))
ratings<-ratingdat[which(ratingdat$N>=15),]

# make actor into an ordered factor, ordering by mean rating
ratings$actor_1_name <- factor(ratings$actor_1_name)
ratings$actor_1_name <- reorder(ratings$actor_1_name, ratings$M)

ratings$actor_1_name


# -> top 30 most liked actors (top 2%)

fdata_2$STARS <- ifelse(fdata_2$primaryName %in% c("Leonardo DiCaprio", "Tom Hanks", "Clint Eastwood",
                                                   "Philip Seymour Hoffman", "Christian Bale", "Harrison Ford",
                                                   "Kevin Spacey", "Denzel Washington","Brad Pitt", "Jennifer Lawrence",
                                                   "Joseph Gordon-Levitt ", "Scarlett Johansson", "Bill Murray","Chris Hemsworth", "Anthony Hopkins",
                                                   "Matt Damon", "Ryan Gosling","Tom Cruise", "Jake Gyllenhaal",
                                                   "Robert Downey Jr.", "Natalie Portmann", "Hugh Jackmann", "Morgan Freeman", "Robert de Niro",
                                                   "Al Pacino", "Keanu Reeves", "Johnny Depp", "Will Smith", "Matthew McConaughey", "Colin Firth"),1,0) # 1 if top 30 otherwise 0


sum(final2$STARS)
# calculate mean rating and SE for each director
ratingdat <- ddply(df, c("director_name"), summarise,
                   M = mean(imdb_score, na.rm=T),
                   SE = sd(imdb_score, na.rm=T)/sqrt(length(na.omit(imdb_score))),
                   N = length(na.omit(imdb_score)))
ratings<-ratingdat[which(ratingdat$N>=10 & !(ratingdat$director_name=='')),]

# make director into an ordered factor, ordering by mean rating:
ratings$director_name <- factor(ratings$director_name)
ratings$director_name <- reorder(ratings$director_name, ratings$M)
ratings$director_name


# -> top 10 most liked directors 

fdata_2$DIREC <- ifelse(fdata_2$primaryName %in% c("David Fincher", "Peter Jackson", "Martin Scorsese",
                                                   "Steven Spielberg", "	Francis Ford Coppola", "Richard Linklater",
                                                   "Robert Zemeckis ", "Clint Eastwood","Stephen Frears", "	Ridley Scott",
                                                   "Rob Reiner", "Woody Allen", "Oliver Stone","	Tim Burton", "Ron Howard"),1,0) # 1 if top 15 otherwise 0



# more cleansing

fdata_3 <- fdata_2 %>% filter(fdata_2$category %in% c("actor", "actress", "director"))
fdata_3 <- select(fdata_3, -c(ordering, genres))

# month

fdata_3$month <- gsub('[0-9]','', fdata_3$`Released Worldwide...3`)
fdata_3$month <- gsub('[-,]','\\',fdata_3$month)
fdata_3$month <- gsub(" ", "", fdata_3$month, fixed = TRUE)

unique(fdata_3$month)

fdata_3$jan <- ifelse(fdata_3$month %in% c("Jan"),1,0)
fdata_3$feb <- ifelse(fdata_3$month %in% c("Feb"),1,0)
fdata_3$mar <- ifelse(fdata_3$month %in% c("Mar"),1,0)
fdata_3$apr <- ifelse(fdata_3$month %in% c("Apr"),1,0)
fdata_3$may <- ifelse(fdata_3$month %in% c("May"),1,0)
fdata_3$jun <- ifelse(fdata_3$month %in% c("Jun"),1,0)
fdata_3$jul <- ifelse(fdata_3$month %in% c("Jul"),1,0)
fdata_3$aug <- ifelse(fdata_3$month %in% c("Aug"),1,0)
fdata_3$sep <- ifelse(fdata_3$month %in% c("Sep"),1,0)
fdata_3$oct <- ifelse(fdata_3$month %in% c("Oct"),1,0)
fdata_3$nov <- ifelse(fdata_3$month %in% c("Nov"),1,0)
fdata_3$dec <- ifelse(fdata_3$month %in% c("Dec"),1,0)

### var names

colnames(fdata_3)[2] <- "domrele" # domestic release
colnames(fdata_3)[3] <- "worldrele" # world release
colnames(fdata_3)[4] <- "domrele_year" # domestic release
colnames(fdata_3)[5] <- "worldrele_year" # world release
colnames(fdata_3)[6] <- "distributor" # world release


### doubles out



small <- with(fdata_3, aggregate(list(STARS = STARS, DIREC=DIREC), list(Title = tolower(Title)), sum))

small$STARS[small$STARS>=1]=1
small$DIREC[small$DIREC>=1]=1

small1 <- small[-1]


last <- fdata_3[!duplicated(fdata_3$Title), ]

last <- last[order(last$Title),]



# final dataset

last1 <- select(last, -c(primaryName:category, STARS, DIREC))

final <- cbind(last1,small1)

final <- final[,c(1 ,12, 24,25, 50, 51 ,7,2,  3,  4,  5,  6,    8,  9, 10, 11,  13, 14, 15,
         16, 17, 18, 19, 20, 21, 22, 23,  26, 27, 28, 29, 30,
         31 ,32 ,33 ,34 ,35, 36, 38,  39, 40, 41, 42, 43,37, 44,45, 46,
         47, 48, 49)]


# dummy hot encode
library(caret)

final$id <- seq.int(nrow(final))

dummie <- select(final, id, Source, `Production Method`, `Creative Type`)
dmy <- dummyVars(" ~ .", data = dummie)
trsf <- data.frame(predict(dmy, newdata = dummie))

final2 <- merge(final, trsf, by ="id", all = FALSE, sort = FALSE)


# export data

library(openxlsx)


write.csv(x=final2, file="final2")

write.xlsx(final2, 'final2.xlsx')

############### no golden globes#################
# golden globes

gold <- read.csv('/Users/Moritz/Desktop/dataset/goldeng.csv', header=TRUE)

gold <- select(gold, -c(year_film:category, film))

colnames(gold)[1] <- "primaryName"

mdf <- merge(gold, fdata_3, by ="primaryName", all = FALSE, sort = FALSE)

length(unique(fdata_3$Title))




#






