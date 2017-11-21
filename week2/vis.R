# general visualisation
library('ggplot2') # visualisation
library('scales') # visualisation
library('grid') # visualisation
library('gridExtra') # visualisation
library('RColorBrewer') # visualisation
library('corrplot') # visualisation

# general data manipulation
library('dplyr') # data manipulation
library('readr') # input/output
library('data.table') # data manipulation
library('tibble') # data wrangling
library('tidyr') # data wrangling
library('stringr') # string manipulation
library('forcats') # factor manipulation
library('magrittr')

# Dates
library('lubridate') # date and time

# Extra vis
library('ggforce') # visualisation
library('ggridges') # visualisation


# Define multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

# function to extract binomial confidence levels
get_binCI <- function(x,n) as.list(setNames(binom.test(x,n)$conf.int, c("lwr", "upr")))

train <- as.tibble(fread('./final.csv'))
getwd()
summary(train)
glimpse(train)
sum(is.na(train))



train <- train %>%
  mutate(state = factor(STFIPS),
         #gender = factor(gender),
         daywait = factor(DAYWAIT),
         year = factor(YEAR))

p1 <- train %>%
  ggplot(aes(GENDER>0, fill = GENDER)) +
  geom_bar() +
  theme(legend.position = "none")
layout <- matrix(c(1,1,2,2,3,3,4,4,4,5,5,5),2,6,byrow=TRUE)
multiplot(p1, p2, p3, p4, p5, layout=layout)

p1 <- train %>%
  ggplot(aes(bd, reorder(reg_via, -bd, FUN = median), fill = reg_via)) +
  geom_density_ridges() +
  geom_vline(xintercept = c(19,25), linetype = 2) +
  theme(legend.position = "none") +
  labs(x = "YEAR", y = "Waiting time") +
  #ggtitle("Age / city / registration method")


p2 <- train %>%
  filter(bd < 65) %>%
  ggplot(aes(reorder(city, -bd, FUN = median), bd, fill = city)) +
  geom_boxplot() +
  theme(legend.position = "none") +
  labs(x = "State", y = "Waiting time") +
  coord_flip()

p3 <- train  %>%
  group_by(city, reg_via) %>%
  summarise(median_age = median(bd)) %>%
  ggplot(aes(city, reg_via, fill = median_age)) +
  geom_tile() +
  labs(x = "City", y = "Registration method", fill = "Median age") +
  scale_fill_distiller(palette = "Spectral")


layout <- matrix(c(1,2,1,2,1,2,3,3,3,3),5,2,byrow=TRUE)
multiplot(p1, p2, p3, layout=layout)
