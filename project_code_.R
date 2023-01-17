setwd("/Users/jonathansimonsen/Desktop/DTU/02445 Project in Statistical Evaluation for AI/Project work/")

data <- read.table('horse_data23.txt', header = TRUE)

# Converting certain applicable attributes to factors 
data$experiment <- as.factor(data$experiment)
data$horse <- as.factor(data$horse)
data$lameLeg <- as.factor(data$lameLeg)
data$lameSide <- as.factor(data$lameSide)
data$lameForeHind <- as.factor(data$lameForeHind)

# Q1: Assessing if horse has a significant effect on either symmetry score using ANOVA
L1 <- lm(S ~ horse, data = data) 
L2 <- lm(A ~ horse, data = data) 
L3 <- lm(W ~ horse, data = data) 

anova(L1)
anova(L2)
anova(L3)

# Q1: Using Kruskal-Wallis test: 
kruskal.test(S ~ horse, data=data)
kruskal.test(A ~ horse, data=data)
kruskal.test(W ~ horse, data=data)

# Histogram to check class distrubition 
counts <- table(data$lameLeg)
barplot(counts,
        xlab="lameLeg", col = viridis(9))


# Histogram to check class distrubition 
counts <- table(data$horse)
barplot(counts,
        xlab="Horse", col = viridis(12))

ggplot(data, aes(fill=lameLeg, x=horse)) + geom_bar(position="stack", stat="identity")

# Boxplots
boxplot(S ~ horse, data=data)


ggplot(data, aes(x=horse, y=A, fill=horse)) + 
  geom_boxplot(alpha=0.6) + 
  xlab("Horse") + theme(legend.position="none")


# QQ-plots 
residualsA <- lm(A ~ horse, data = data)$residuals
qqnorm(residualsA, main="A")
qqline(residualsA, col="green")

residualsW <- lm(W ~ horse, data = data)$residuals
qqnorm(residualsW, main="W")
qqline(residualsW, col="blue")

residualsS <- lm(S ~ horse, data = data)$residuals
qqnorm(residualsS, main="S")
qqline(residualsS, col="red")


# bar chart 
M <- table(data$lameLeg, data$horse)
barplot(M, main = "Distribution of lameness per horse", xlab = "horse", ylab = "lameLeg",
        col = alpha("white", 0))

barplot()

legend("topright",
       legend = rownames(M),
       pch = 15, col = viridis(5), cex=1)


chisq.test(data$horse, data$lameLeg)



