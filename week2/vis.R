su
p1 <- train %>%
  ggplot(aes(DAYWAIT, fill = DAYWAIT)) +
  geom_bar() +
  theme(legend.position = "none")
                    
p2 <- train %>%
  ggplot(aes(GENDER, fill = GENDER)) +
  geom_bar() +
  theme(legend.position = "none")

layout <- matrix(c(1,1,2,2,3,3,4,4,4,5,5,5),2,6,byrow=TRUE)
multiplot(p1, p2, layout=layout)

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



sum(train$DAYWAIT < 0); percent(sum(train$DAYWAIT < 0) / sum(train$DAYWAIT >= -100))
sum(train$GENDER < 0); percent(sum(train$GENDER < 0) / sum(train$GENDER >= -100))
sum(train$RACE < 0); percent(sum(train$RACE < 0) / sum(train$RACE >= -100))
sum(train$ETHNIC < 0); percent(sum(train$ETHNIC < 0) / sum(train$ETHNIC >= -100))
sum(train$MARSTAT < 0); percent(sum(train$MARSTAT < 0) / sum(train$MARSTAT >= -100))
sum(train$PREG < 0); percent(sum(train$PREG < 0) / sum(train$PREG >= -100))

sum(train$SUB1 < 0); percent(sum(train$SUB1 < 0) / sum(train$SUB1 >= -100))
sum(train$VET < 0); percent(sum(train$VET < 0) / sum(train$VET >= -100))
sum(train$EMPLOY < 0); percent(sum(train$EMPLOY < 0) / sum(train$EMPLOY >= -100))
sum(train$EDUC < 0); percent(sum(train$EDUC < 0) / sum(train$EDUC >= -100))


# library('plyr')
# fun <- function(x){
#   percent(sum(trainx< 0) / sum(x >= -100))
# }   
# colwise(fun)(train)