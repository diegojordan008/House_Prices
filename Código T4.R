rm(list=ls())

library(xgboost)
library(dplyr)
library(tm)
library(ggplot2)
library(GGally)

setwd("C:/Users/amali/Documents/DiTella/Introducción a Data Science/Talleres/Taller 4")

one_hot_sparse <- function(data_set) {
  
  # IMPORTANTE: si una de las variables es de fecha, la va a ignorar
  
  require(Matrix)
  created <- FALSE
  
  if (sum(sapply(data_set, is.numeric)) > 0) {  # Si hay, Pasamos los numéricos a una matriz esparsa (sería raro que no estuviese, porque "Price"  es numérica y tiene que estar sí o sí)
    out_put_data <- as(as.matrix(data_set[,sapply(data_set, is.numeric)]), "dgCMatrix")
    created <- TRUE
  }
  
  if (sum(sapply(data_set, is.logical)) > 0) {  # Si hay, pasamos los lógicos a esparsa y lo unimos con la matriz anterior
    if (created) {
      out_put_data <- cbind2(out_put_data,
                             as(as.matrix(data_set[,sapply(data_set, is.logical)]), "dgCMatrix"))
    } else {
      out_put_data <- as(as.matrix(data_set[,sapply(data_set, is.logical)]), "dgCMatrix")
      created <- TRUE
    }
  }
  
  # Identificamos las columnas que son factor (OJO: el data.frame no debería tener character)
  fact_variables <- names(which(sapply(data_set, is.factor)))
  
  # Para cada columna factor hago one hot encoding
  i <- 0
  
  for (f_var in fact_variables) {
    
    f_col_names <- levels(data_set[[f_var]])
    f_col_names <- gsub(" ", ".", paste(f_var, f_col_names, sep = "_"))
    j_values <- as.numeric(data_set[[f_var]])  # Se pone como valor de j, el valor del nivel del factor
    
    if (sum(is.na(j_values)) > 0) {  # En categóricas, trato a NA como una categoría más
      j_values[is.na(j_values)] <- length(f_col_names) + 1
      f_col_names <- c(f_col_names, paste(f_var, "NA", sep = "_"))
    }
    
    if (i == 0) {
      fact_data <- sparseMatrix(i = c(1:nrow(data_set)), j = j_values,
                                x = rep(1, nrow(data_set)),
                                dims = c(nrow(data_set), length(f_col_names)))
      fact_data@Dimnames[[2]] <- f_col_names
    } else {
      fact_data_tmp <- sparseMatrix(i = c(1:nrow(data_set)), j = j_values,
                                    x = rep(1, nrow(data_set)),
                                    dims = c(nrow(data_set), length(f_col_names)))
      fact_data_tmp@Dimnames[[2]] <- f_col_names
      fact_data <- cbind(fact_data, fact_data_tmp)
    }
    
    i <- i + 1
  }
  
  if (length(fact_variables) > 0) {
    if (created) {
      out_put_data <- cbind(out_put_data, fact_data)
    } else {
      out_put_data <- fact_data
      created <- TRUE
    }
  }
  return(out_put_data)
}

calculate_mode <- function(x) {
  uniqx <- unique(na.omit(x))
  uniqx[which.max(tabulate(match(x, uniqx)))]
}

## Cargo los datos

data_set <- readRDS("training_set.RDS")

##Análisis exploratorio

data_set$price <- log(data_set$price + 1)  # Voy a predecir el logaritmo de los precios

correlacion <- ggcorr(data_set[5:20], label = TRUE, label_alpha = 0.5)

ggsave("correlograma.png", correlacion)

sapply(data_set, function(data_set) sum(is.na(data_set)))

write.csv(sapply(data_set, function(data_set) sum(is.na(data_set))), "tablaNAs.csv", sep = ",")

preciocuartos <- ggplot(data_set) + geom_point(mapping = aes(bedrooms, price), color = "red") + ylim(0, 1e+07) + xlim(0,15) +
    xlab("Cuartos") + ylab("Precio") + ggtitle("Precio de propiedades dependiente de cantidad de cuartos" )+
    theme_bw()

ggsave("preciocuartos.png", preciocuartos)

ggplot(data_set) + geom_smooth(mapping = aes(created_on, price))

ggsave("creacion.png",ggplot(data_set) + geom_smooth(mapping = aes(created_on, price)))

## Genero nuevas variables, limpio lo que haga falta

data_set <- data_set %>% group_by(l2) %>% mutate(l3 = if_else(is.na(l3), calculate_mode(l3), l3)) %>% ungroup()

data_set$l3 <- (ifelse(is.na(data_set$l3), as.character(data_set$l2), as.character(data_set$l3)))

data_set <- data_set %>% group_by(l3) %>% mutate(surface_total = ifelse(is.na(surface_total), as.integer(mean(surface_total, na.rm = TRUE)), as.integer(surface_total)))

data_set <- data_set %>% group_by(l3) %>% mutate(surface_covered = ifelse(is.na(surface_covered), as.integer(mean(surface_covered, na.rm = TRUE)), as.integer(surface_covered))) %>% ungroup()

data_set <- data_set %>% group_by(surface_covered) %>% mutate(bathrooms = ifelse(is.na(bathrooms), as.integer(mean(bathrooms, na.rm = TRUE)), as.integer(bathrooms))) %>% ungroup()

data_set <- data_set %>% group_by(surface_covered) %>% mutate(bedrooms = ifelse(is.na(bedrooms), as.integer(mean(bedrooms, na.rm = TRUE)), as.integer(bedrooms))) %>% ungroup()

## Hago bow model

corpus <- VCorpus(VectorSource(data_set$description))



corpus <- tm_map(corpus, content_transformer(tolower))



corpus <- tm_map(corpus, content_transformer(removePunctuation))

stp_words <- stopwords("spanish")[stopwords("spanish") != "no"]
corpus <- tm_map(corpus, content_transformer(function(x) removeWords(x, stp_words)))

dt.mat <- DocumentTermMatrix(corpus,
                             
                             control=list(stopwords=FALSE,
                                          
                                          wordLengths=c(1, Inf),
                                          
                                          bounds=list(global=c(25,Inf))))



X_bow <- Matrix::sparseMatrix(i=dt.mat$i, 
                              
                              j=dt.mat$j, 
                              
                              x=dt.mat$v, 
                              
                              dims=c(dt.mat$nrow, dt.mat$ncol),
                              
                              dimnames = dt.mat$dimnames)



corpus_title <- VCorpus(VectorSource(data_set$title))



corpus_title <- tm_map(corpus_title, content_transformer(tolower))



corpus_title <- tm_map(corpus_title, content_transformer(removePunctuation))

corpus_title <- tm_map(corpus_title, content_transformer(function(x) removeWords(x, stp_words)))



dt.mat_title <- DocumentTermMatrix(corpus_title,
                                   
                                   control=list(stopwords=FALSE,
                                                
                                                wordLengths=c(1, Inf),
                                                
                                                bounds=list(global=c(25,Inf))))



X_bow_title <- Matrix::sparseMatrix(i=dt.mat_title$i, 
                                    
                                    j=dt.mat_title$j, 
                                    
                                    x=dt.mat_title$v, 
                                    
                                    dims=c(dt.mat_title$nrow, dt.mat_title$ncol),
                                    
                                    dimnames = dt.mat_title$dimnames)


## Hago one-hot-encoding

one_hot_training_set <- one_hot_sparse(data_set %>% select(-title, -description, -l6, -l5, -l4, -currency))

one_hot_training_set <- cbind(one_hot_training_set, X_bow, X_bow_title)

rm(dt.mat, X_bow, X_bow_title)

## Separo el conjunto de evaluación

train_set <- one_hot_training_set[data_set$created_on < "2019-08-01",]
eval_set <- one_hot_training_set[data_set$created_on >= "2019-08-01",]

valid_index <- sample(1:nrow(train_set), round(0.02 * nrow(train_set)))
training_set <- train_set[setdiff(1:nrow(train_set), valid_index),]
validation_set <- train_set[valid_index,]

random_grid <- function(size,
                        min_nrounds, max_nrounds,
                        min_max_depth, max_max_depth,
                        min_eta, max_eta,
                        min_gamma, max_gamma,
                        min_colsample_bytree, max_colsample_bytree,
                        min_min_child_weight, max_min_child_weight,
                        min_subsample, max_subsample) {
  
  rgrid <- data.frame(nrounds = sample(c(min_nrounds:max_nrounds),
                                       size = size, replace = TRUE),
                      max_depth = sample(c(min_max_depth:max_max_depth),
                                         size = size, replace = TRUE),
                      eta = round(runif(size, min_eta, max_eta), 5),
                      gamma = round(runif(size, min_gamma, max_gamma), 5),
                      colsample_bytree = round(runif(size, min_colsample_bytree,
                                                     max_colsample_bytree), 5),
                      min_child_weight = round(runif(size, min_min_child_weight,
                                                     max_min_child_weight), 5),
                      subsample = round(runif(size, min_subsample, max_subsample), 5))
  return(rgrid)    
}

train_xgboost <- function(data_train, data_val, rgrid) {
  
  watchlist <- list(train = data_train, valid = data_val)
  
  predicted_models <- list()
  
  for (i in seq_len(nrow(rgrid))) {
    print(i)
    print(rgrid[i,])
    trained_model <- xgb.train(data = data_train,
                               params=as.list(rgrid[i, c("max_depth",
                                                         "eta",
                                                         "gamma",
                                                         "colsample_bytree",
                                                         "subsample",
                                                         "min_child_weight")]),
                               nrounds = rgrid[i, "nrounds"],
                               watchlist = watchlist,
                               objective = "reg:squarederror",
                               eval.metric = "rmse",
                               print_every_n = 10)
    
    perf_tr <- tail(trained_model$evaluation_log, 1)$train_rmse
    perf_vd <- tail(trained_model$evaluation_log, 1)$valid_rmse
    print(c(perf_tr, perf_vd))
    
    predicted_models[[i]] <- list(results = data.frame(rgrid[i,],
                                                       perf_tr = perf_tr,
                                                       perf_vd = perf_vd),
                                  model = trained_model)
    rm(trained_model)
    gc()
  }
  
  return(predicted_models)
}


result_table <- function(pred_models) {
  res_table <- data.frame()
  i <- 1
  for (m in pred_models) {
    res_table <- rbind(res_table, data.frame(i = i, m$results))
    i <- i + 1
  }
  res_table <- res_table[order(-res_table$perf_vd),]
  return(res_table)
}

rgrid <- random_grid(size = 25,
                     min_nrounds = 100, max_nrounds = 500,
                     min_max_depth = 7, max_max_depth = 25,
                     min_eta = 0.01, max_eta = 0.07,
                     min_gamma = 0, max_gamma = 0.3,
                     min_colsample_bytree = 0.3, max_colsample_bytree = 0.5,
                     min_min_child_weight = 0, max_min_child_weight = 3,
                     min_subsample = 0.3, max_subsample = 0.5)

dtrain <- xgb.DMatrix(data = training_set[,setdiff(colnames(training_set), "price")],
                      label = training_set[,"price"])
dvalid <- xgb.DMatrix(data = validation_set[,setdiff(colnames(validation_set), "price")],
                      label = validation_set[,"price"])

predicted_models <- train_xgboost(dtrain, dvalid, rgrid)

results <- result_table(predicted_models)

base_model <- xgb.train(data = dtrain,
                        params = list(max_depth = 24,
                                      eta = 0.05465, 
                                      gamma = 0.09920,
                                      colsample_bytree = 0.45186,
                                      subsample = 0.48034,
                                      min_child_weight = 0.61845),
                        nrounds = 1400,
                        watchlist = list(train = dtrain, valid = dvalid),
                        objective = "reg:squarederror",
                        eval.metric = "rmse",
                        print_every_n = 10)

deval <- xgb.DMatrix(data = eval_set[,setdiff(colnames(eval_set), "price")])
predicciones <- exp(predict(base_model, newdata=deval))-1

predicciones <- data.frame(id=data_set[data_set$created_on >= "2019-08-01", "id"],
                           price=predicciones)

write.table(predicciones, "predicciones_15.txt", sep=",",
            row.names=FALSE, quote=FALSE)

importanciavar <- xgb.importance(model = base_model)
