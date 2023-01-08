################################################################################
#        Disease progression model anchored around clinical diagnosis:         #
#                Specification of the model with Stan language                 #
#                                                                              #
# Lespinasse et al. Disease progression model anchored around clinical         #
# diagnosis in longitudinal cohorts: example of Alzheimer’s disease and        #
# related dementia.                                                            #
# See : https://github.com/jrmie/dpm_anchored                                  #
################################################################################


# Functions block ---------------------------------------------------------------
functions{
    // This function is used by the map_rect() function to parallel computing
    vector map_ranef(vector phi, vector theta, real[] xr, int[] n_obs_out) {

        # data ----------------------------------------------------------------
        int n_obs_ = n_obs_out[1];
        int n_out_ = n_obs_out[2];
        int n_x_ = n_obs_out[3];
        int outcome_[n_obs_] = n_obs_out[4:(3+n_obs_)];
        matrix[n_obs_, n_x_] x_ = to_matrix(xr[2:(1+n_x_*n_obs_)], n_obs_, n_x_);
        vector[n_obs_] y_ = to_vector(xr[(1+n_x_*n_obs_+1):(1+n_obs_*(n_x_+1))]);
        real time_[n_obs_] = xr[(1+n_obs_*(n_x_+1)+1):(1+n_obs_*(n_x_+2))];
        real t_evt_ = xr[1];

        # population parameters ------------------------------------------------
        matrix[n_out_, n_x_] beta_ = to_matrix(phi[1:(n_out_*n_x_)], n_out_, n_x_);
        vector[n_out_] gamma_ = phi[(n_out_*n_x_+1) : ((n_x_+1)*n_out_)];
        vector[n_out_] sigma_e_ = phi[((n_x_+1)*n_out_+1) : ((n_x_+2)*n_out_)];

        # individual parameters ------------------------------------------------
        vector[n_out_] z_u0_ = to_vector(theta[1:n_out_]);
        vector[n_out_] z_u1_ = to_vector(theta[(n_out_+1):(2*n_out_)]);
        real T_dem_i_ = theta[(2*n_out_)+1];

        # model ----------------------------------------------------------------
        vector[n_obs_] mu;                              // mean of the predictor
        vector[n_obs_] sigma_e_aug;              // variance of the random error
        real ll;                                                   // likelihood

        for (i in 1:n_obs_) {
            mu[i] = dot_product(x_[i], beta_[outcome_[i]]) 
                    // sum of the outcome specific fixed effects for covariates 
                    // and intercepts
                    + gamma_[outcome_[i]] * (time_[i]-T_dem_i_) 
                    // fixed outcome specific slope
                    + z_u0_[outcome_[i]]
                    // outcome and individual specific random intercept
                    + z_u1_[outcome_[i]]*(time_[i]-T_dem_i_);
                    // outcome and individual specific random slope
            sigma_e_aug[i] = sigma_e_[outcome_[i]];
            // outcome specific variance error
        }
        
        // return the log probability density of the observations y_jk for 
        // individual i given mu and variance error
        ll = normal_lpdf(y_ | mu, sigma_e_aug);
        return[ll]';
    }
}


# Data block -------------------------------------------------------------------
data{
    int<lower = 0> n_obs;                              // number of observations
    int<lower = 0, upper = n_obs> n_sub;                   // number of subjects
    int<lower = 0, upper = n_obs> n_out;                   // number of outcomes
    int<lower = 0> n_x;            // number of covariates (including intercept)
    matrix[n_obs, n_x] x;                                   // covariates values
    vector[n_obs] y;                                           // outcome values
    vector[n_obs] time;                                  // times at occasions j
    int<lower = 1, upper = n_out> outcome[n_obs];       // indicator for outcome
    int<lower = 1, upper = n_sub> subject[n_obs];       // indicator for subject
    int<lower = 1, upper = n_obs> sub_n_obs[n_sub];    // number of observations
                                                       //       for each subject
    int<lower = 0, upper = n_sub> n_ndem;     // number of subjects non demented
    vector[n_sub] dem;                                 // indicator of diagnosis
    vector[n_sub] t_evt;               // time at event (censoring or diagnosis)
    int id_slp[n_out];                              // indicator of random slope
    real C;    // constraint value which define uncertainty about diagnosis time
}


# Transformations block applied to data ----------------------------------------
transformed data{
    vector[n_sub] L_eta;     // individual constrained lower value for eta_T_dem  
    vector[n_sub] U_eta;     // individual constrained upper value for eta_T_dem 
    real xr[n_sub, (n_x+2)*max(sub_n_obs) + 1];
    // individual specific array of real data [x, y, time, t_evt]
    int n_obs_out[n_sub, 3 + max(sub_n_obs)];
    // individual specific array of integer data [n_obs, n_out, n_x, outcome]
    
    int end;
    int pos = 1;
    for (i_sub in 1:n_sub) {
      
      // This part split individual specific real and integer data types into 
      // arrays for parallel computing. Length is the maximum number of 
      // observations for a single individual and fill with zeros if current 
      // individual have less observations.
      int n = sub_n_obs[i_sub];
      int diff = max(sub_n_obs)-n;
      int fill_int = 0;
      real fill_real = 0;
      end = pos + n - 1;
      n_obs_out[i_sub] = append_array(append_array({n, n_out, n_x},
                                                    outcome[pos:end]),
                                                    rep_array(fill_int, diff));
      xr[i_sub] = append_array(append_array({t_evt[i_sub]},
                                             to_array_1d(append_col(x[pos:end,],
                                             append_col(y[pos:end],
                                                        time[pos:end])))),
                                             rep_array(fill_real,
                                                       diff*(n_x+2)));
      pos += n;

      // This part define lower and upper bounds for individual eta_T_dem 
      // parameter according to the type and time of event. Log transformation
      // is applied as eta_T_dem is in the log scale
      L_eta[i_sub] = log(t_evt[i_sub]+0.0001); // .0001 avoid infinite value
      if (dem[i_sub] == 1){                    
        U_eta[i_sub] = log(t_evt[i_sub]+(2*C));
      } else {
        U_eta[i_sub] = log(100);
        // 100 years is used as upper bound in order not to  slow down the 
        // sampling with inconsistent values of time reparametrisation.
      }
    }
}


# Parameters block -------------------------------------------------------------
parameters{
    matrix[n_out, n_x] beta;                
    // outcome specific fixed intercepts and covariates effects
    vector<lower = 0>[n_out] gamma;             // outcome specific fixed slopes
    vector<lower = 0>[n_out] sigma_e;     // outcome specific variances of error
    
    vector<lower = 0>[n_out] sigma_u0;         // variances of random intercepts
    vector<lower = 0>[sum(id_slp)] sigma_u1;       // variances of random slopes
    matrix[n_sub, n_out] eta_u0;             // raw individual random intercepts
    matrix[n_sub, sum(id_slp)] eta_u1;           // raw individual random slopes
    
    real<lower = 0> mu_t_dem;                     // mean of eta_t_dem parameter
    real<lower = 0> sigma_t_dem;              // variance of eta_t_dem parameter
    vector<lower = 0, upper = 1>[n_sub] eta_t_dem_raw; 
    // raw individual parameter of time reparametrisation
}


# Transformation block applied to parameters -----------------------------------
transformed parameters{
    vector[n_out*n_x + 2*n_out] phi;
    // array of population parameters [beta + gamma + sigma_e]
    vector[2*n_out + 1] theta[n_sub];
    // array of individual specific parameters [z_u0 + z_u1 + T_dem_i]
    
    matrix[n_sub, n_out] z_u0;                   // individual random intercepts
    matrix[n_sub, n_out] z_u1;                       // individual random slopes
    
    // eta_t_dem si the individual time shift in the log scale of the time
    vector<offset = mu_t_dem, 
           multiplier = sigma_t_dem>[n_sub] eta_t_dem = L_eta
                                                      + (U_eta-L_eta)
                                                      .* eta_t_dem_raw;
    // T_dem_i is the individual time shift in the observed time scale
    vector[n_sub] T_dem_i = exp(eta_t_dem) - C;
    
    {
    int j_slp = 1;
    for (i in 1:n_out){
      z_u0[,i] = eta_u0[,i]*sigma_u0[i];
      if (id_slp[i] == 0){
        z_u1[,i] = rep_vector(0, n_sub);
      } else {
        z_u1[,i] = eta_u1[,j_slp]*sigma_u1[j_slp];
        j_slp += 1;
      }
    }
    }

    {
    phi[1:n_out*n_x] = to_vector(beta);
    phi[n_out*n_x+1 : (n_x+1)*n_out] = gamma;
    phi[(n_x+1)*n_out+1 : (n_x+2)*n_out] = sigma_e;
    for (t in 1:n_sub) {
      theta[t] = append_col(z_u0[t,], append_col(z_u1[t,], T_dem_i[t]))';
      }
    }
}


# Model block ------------------------------------------------------------------
model{

    // Priors of the parameters
    to_vector(beta) ~ normal(0, 10);
    
    for (i in 1:n_out) {
      gamma[i] ~ normal(0, 10);
      sigma_e[i] ~ cauchy(0, 2.5);
      sigma_u0[i] ~ cauchy(0, 2.5);
      to_vector(eta_u0[,i]) ~ normal(0, 1);
    }
    
    for (i in 1:sum(id_slp)) {
     sigma_u1[i] ~ cauchy(0, 2.5);
     to_vector(eta_u1[,i]) ~ normal(0, 1);
    }

    mu_t_dem ~ normal(10, 10);
    sigma_t_dem ~ normal(0, 2.5);
    eta_t_dem ~ normal(mu_t_dem, sigma_t_dem); // imply that T_dem_i ~ lognormal()

    // likelihood
    target += sum(map_rect(map_ranef, phi, theta, xr, n_obs_out));
      // map_rect() function is implemented in stan software to parallel 
      // computation
}


# Posterior likelihood and prediction for each observation ---------------------
generated quantities{
  vector[n_obs] log_lik;
  vector[n_obs] preds;

  // log-likelihood and preds
  for (i in 1:n_obs) {
      log_lik[i] = normal_lpdf(y[i] | dot_product(x[i], beta[outcome[i]]) 
                + gamma[outcome[i]] * (time[i]-T_dem_i[subject[i]])
                + z_u0[subject[i], outcome[i]]
                + z_u1[subject[i], outcome[i]]*(time[i]-T_dem_i[subject[i]]),
                sigma_e[outcome[i]]);
      preds[i] = dot_product(x[i], beta[outcome[i]])
                + gamma[outcome[i]] * (time[i]-T_dem_i[subject[i]])
                + z_u0[subject[i], outcome[i]]
                + z_u1[subject[i], outcome[i]]*(time[i]-T_dem_i[subject[i]]);
      }
}
