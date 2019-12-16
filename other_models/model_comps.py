from matplotlib import pyplot as plt

# Creates figure showing root mean squared error differences between our model and other models
errors = [347.9622447104464, 341.2779815235158, 453.52849382082593, 5985.578839675354]
model_names = ['RF', 'Pre-built RF', 'Linear Reg', 'Gradient Boost']
plt.title('BTC Prediction Error by Model')
plt.xlabel('Models')
plt.ylabel('Root Mean Squared Error')
plt.scatter(model_names, errors, c='black')
plt.savefig('../figures/modelcomps.png')
plt.show()
