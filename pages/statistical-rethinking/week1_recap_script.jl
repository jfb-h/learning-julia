using CairoMakie
using Distributions
using StatsBase

## Simulate the experiment

function simulate_globe(N, p) 
    outcomes = ["W", "L"]
    probabilities = [p, 1-p]
    results = sample(outcomes, Weights(probabilities), N)
    count(==("W"), results)
end

simulate_globe_binomial(N, p) = rand(Binomial(N, p))

## Run the experiment

N = 100
p_true = 0.7
data = simulate_globe(N, p_true)


# Prior distribution

prior(p) = pdf(Beta(3,2), p)

function plot_beta()
    fig = Figure()
    ax = Axis(fig[1,1])
    plot!(ax, Beta(1,1); label="Beta(1,1)", color=:blue)
    plot!(ax, Beta(2,2); label="Beta(2,2)", color=:red)
    plot!(ax, Beta(3,2); label="Beta(3,2)", color=:green)
    axislegend(ax)
    fig
end

plot_beta()

# prior predictive

function predictive(N, prior_or_posterior; S=100000)
    samples = rand(prior_or_posterior, S)
    [simulate_globe(N, p) for p in samples]
end

prior_predictions = predictive(N, Beta(3,2))
hist(prior_predictions; color=:grey80, strokewidth=1)


# likelihood

likelihood(p; y=data, N=N) = pdf(Binomial(N, p), y)

lines(0:0.01:1, likelihood)

# joint

joint(p) = prior(p) * likelihood(p)

lines(0:0.01:1, joint)


# analytical posterior

analytical_posterior(N, y; α=3, β=2) = Beta(α + y, β + N - y)

# grid approximation

grid = 0:0.001:1

grid_joint = joint.(grid)
posterior = grid_joint ./ sum(grid_joint)
lines(grid, posterior)

posterior_samples = sample(grid, Weights(posterior), 10_000)
hist(posterior_samples); xlims!(0, 1); current_figure()


# posterior predictive distribution

posterior_predictive = [simulate_globe(N, p) for p in posterior_samples]
hist(posterior_predictive); xlims!(0, 100); current_figure()

hist(predictive(N, analytical_posterior(N, data)))