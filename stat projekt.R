
data <- horse_data23


#converter data til factors
data$experiment <- as.factor(data$experiment)

data$horse <- as.factor(data$horse)
data$lameLeg <- as.factor(data$lameLeg)
data$lameSide <- as.factor(data$lameSide)
data$lameForeHind <- as.factor(data$lameForeHind)


#collapsed data

data$collapsed <- data$lameLeg

data$collapsed[data$collapsed == "right:fore"] <- "right:left"
data$collapsed[data$collapsed == "left:hind"] <- "right:left"
data$collapsed[data$collapsed == "left:fore"] <- "left:right"
data$collapsed[data$collapsed == "right:hind"] <- "left:right"
# analyse af sammenhæng mellem A og horse

L <- lm(A ~ horse, data = data)


boxplot(A ~ horse,data = data )
summary(L)

anova(L)

# analyse af sammenhæng mellem S og horse

L <- lm(S ~ horse, data = data)


boxplot(S ~ horse,data = data) 
summary(L)

anova(L)


# analyse af sammenhæng mellem W og horse

L <- lm(W ~ horse, data = data)


boxplot(W ~ horse,data = data )
summary(L)

anova(L)

##########################################

#undersøgelse af variabler

# A og W

plot(data$W,data$A)


hist(data$W,breaks= 15)
hist(data$A,breaks = 15)
#der ser ud til at være lineær sammenhæng

cor(data$W,data$A)

#undersøgelse af pc3/pc4

plot(data$pc3,data$pc4)

#umiddelbar mindre lineær sammenhæng mellem pc3 og pc4
cor(data$pc3,data$pc4)

hist(data$pc3,breaks= 15)
hist(data$pc4,breaks = 15)

#de er nogenlunde normalfordelt.


# bare lige et forsøg på lineært at classificere mellem lameness og ikke lameness


data$lame <-(data$lameLeg != "none")*1




train_ids <- 1:65

lame_train <- data[train_ids, ]
lame_test <- data[- train_ids, ]


L <- glm(lame ~ W + A,data = lame_train,family = binomial)


probabilities <- predict(L,lame_test,type = "response")
predicted.classes <- ifelse(probabilities > 0.5, 1, 0)
# Model accuracy
mean(predicted.classes == lame_test$lame)

sum(data$lame == 1)
# den predicter umiddelbart bare altid lame hvilket giver 85% accuracy på test splittet


L <- glm(lame ~ W * A,data = data,family = binomial)
summary(L)


plot(data$A,data$W,col=data$lameLeg,pch=16,xlab = "A",ylab = "W",main = "scatter plot of A and W")

legend("topleft", inset=.02, title="lameness",
       c("none", "right:hind","right:fore","left:hind","left:fore"),fill = data$lameLeg, horiz=FALSE, cex=0.8)



plot(data$pc3,data$pc4,col=colors2[data$lameLeg],pch=16,xlab = "pc3",ylab = "pc4",main = "scatter plot of pc3 and pc4")

legend("topleft", legend = levels(factor(data$lameLeg)), pch = 19, col = )

ggplot(data,aes(x=A,y=W,col=lameLeg))+geom_point(size = 3)

ggplot(data,aes(x=pc3,y=pc4,col=lameLeg))+geom_point(size = 3)


ggplot(data,aes(x=pc3,y=pc4,col=collapsed))+geom_point(size = 3)
ggplot(data,aes(x=A,y=W,col=collapsed))+geom_point(size = 3)

library(ggplot2)

# permutation

qqnorm(r$B1, main = "B1")
qqline(r$B1)


r <- split(data$W, data$horse)

mean(r$B1)
mean(r$B2)
mean(r$B3)
mean(r$B4)
mean(r$B5)
mean(r$B6)
mean(r$B7)
mean(r$B9)

#størst forskel mellem 5 og 9

wilcox.test(r$B9,r$B5)


r <- split(data$S, data$horse)

mean(r$B1)
mean(r$B2)
mean(r$B3)
mean(r$B4)
mean(r$B5)
mean(r$B6)
mean(r$B7)
mean(r$B9)

#størst forskel mellem 5 og 9

wilcox.test(r$B6,r$B4)

r <- split(data$A, data$horse)

mean(r$B1)
mean(r$B2)
mean(r$B3)
mean(r$B4)
mean(r$B5)
mean(r$B6)
mean(r$B7)
mean(r$B9)

wilcox.test(r$B9,r$B2)


kruskal.test(W ~ horse,data = data)
kruskal.test(S ~ horse,data = data)
kruskal.test(A ~ horse,data = data)


sum(data$horse == "B3")



p <- c(0.00030,3.8e-05,0.62,0.082,5.1e-05,2.5e-05, 0.024, 0.0059, 0.75, 0.23, 0.020,6.3e-05 , 0.0039, 0.11, 0.99, 0.16, 0.00029, 7.9e-05, 0.29, 0.0068, 0.097)   



p.adjust(p,method = "bonferroni")