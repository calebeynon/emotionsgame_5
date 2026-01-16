# nolint start
library(data.table)
library(ggplot2)

dtr = function(x) as.data.table(read.csv(x))
dt = dtr('datastore/derived/contributions.csv')


dt_t = dt[,.(cont = mean(contribution)), by = .(round,segment)]

dt_t = dt_t[order(segment, round)]
dt_t[, x_seq := 1:.N]
dt_t[, seg_num := as.numeric(gsub("\\D", "", segment))]
dt_t[, x_label := paste0("s", seg_num, "r", round)]
p = ggplot(dt_t, aes(x = x_seq, y = cont, group = segment, color = segment))+
    theme_minimal()+
    theme(panel.grid.minor.x = element_blank())+
    geom_line()+
    geom_point()+
    scale_x_continuous(breaks = dt_t$x_seq, labels = dt_t$x_label)+
    scale_y_continuous(breaks = seq(0,25,1), labels = seq(0,25,1))

ggsave('plots/all/contributions_by_segment_round.png',p,width = 6.5,height = 4, units = 'in')

dt_t = dt[,.(cont = mean(contribution)), by = .(round,segment,treatment)]

dt_t = dt_t[order(segment, round)]
dt_t[, seg_round_key := paste(segment, round, sep = "_")]
unique_keys = unique(dt_t[, .(seg_round_key, segment, round)][order(segment, round)])
unique_keys[, x_seq := 1:.N]
dt_t = merge(dt_t, unique_keys[, .(seg_round_key, x_seq)], by = "seg_round_key")
dt_t[, seg_num := as.numeric(gsub("\\D", "", segment))]
dt_t[, x_label := paste0("s", seg_num, "r", round)]
axis_breaks = unique(dt_t[, .(x_seq, x_label)])[order(x_seq)]

dt_t[, group_var := paste(treatment, segment, sep = "_")]

p = ggplot(dt_t, aes(x = x_seq, y = cont, group = group_var, color = factor(treatment)))+
    theme_minimal()+
    theme(panel.grid.minor.x = element_blank())+
    geom_line()+
    geom_point()+
    scale_x_continuous(breaks = axis_breaks$x_seq, labels = axis_breaks$x_label)+
    scale_y_continuous(breaks = seq(0,25,1), labels = seq(0,25,1))+
    labs(color = "Treatment")

ggsave('plots/all/contributions_by_segment_round_treatment.png',p,width = 6.5,height = 4, units = 'in')


