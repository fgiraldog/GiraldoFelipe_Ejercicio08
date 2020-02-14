import numpy as np 
import matplotlib.pyplot as plt 


data = np.genfromtxt('data_to_fit.txt')

x = data[:,0]
y = data[:,1]
sigma_y = data[:,2]

def model_a(x,params):
	y = params[0] + x*params[1] + params[2]*x**2

	return y

def model_b(x,params):
	y = params[0]*(np.exp(-0.5*(x-params[1])**2/params[2]**2))

	return y

def model_c(x,params):
	y = params[0]*(np.exp(-0.5*(x-params[1])**2/params[2]**2))
	y+= params[0]*(np.exp(-0.5*(x-params[3])**2/params[4]**2))

	return y

def log_likelihood(x,y,sigma,params,mod):

	log_like = 0

	if mod == 1:
		for xx,yy,ss in zip(x,y,sigma):
			log_like += -0.5*((((model_a(xx,params))-yy)**2)/(ss**2))

	if mod == 2:
		for xx,yy,ss in zip(x,y,sigma):
			log_like += -0.5*((((model_b(xx,params))-yy)**2)/(ss**2))

	if mod == 3:
		for xx,yy,ss in zip(x,y,sigma):
			log_like += -0.5*((((model_c(xx,params))-yy)**2)/(ss**2))


	return log_like


def MC(x,y,s,pasos,mod,sig_MC):

	if mod == 1:
		betas = [2.,2.,2.]
	if mod == 2:
		betas = [2.,2.,2.]
	if mod == 3:
		betas = [2.,2.,2.,2.,2.]
	
	L_parado = log_likelihood(x,y,s,betas,mod)
	L_camino = L_parado
	betas_camino = betas

	for i in range(0,pasos):
		betas_ale = np.random.normal(betas, sig_MC, size = len(betas))
		L_ale = log_likelihood(x,y,s,betas_ale,mod)
		alpha = np.exp(L_ale-L_parado)

		if alpha >= 1:
			betas_camino = np.vstack([betas_camino,betas_ale])
			L_camino = np.append(L_camino,L_ale)
			betas = betas_ale
			L_parado = L_ale


		else:
			beta = np.random.random()

			if alpha >= beta:
				betas_camino = np.vstack([betas_camino,betas_ale])
				L_camino = np.append(L_camino,L_ale)
				betas = betas_ale
				L_parado = L_ale

			else:
				betas_camino = np.vstack([betas_camino,betas])
				L_camino = np.append(L_camino,L_parado)

	return betas_camino, L_camino


def BIC(params,k,x,y,sigma,mod):
	L = -log_likelihood(x,y,sigma,params,mod)
	L += (k*np.log(len(x)))/(2.)

	return L*2

betas_1, L_1 = MC(x,y,sigma_y,20000,1,0.5)
best_1 = betas_1[np.argmax(L_1)]

betas_2, L_2 = MC(x,y,sigma_y,20000,2,0.1)
best_2 = betas_2[np.argmax(L_2)]

betas_3, L_3 = MC(x,y,sigma_y,20000,3,0.1)
best_3 = betas_3[np.argmax(L_3)]

plt.figure(figsize = (10,8))
for i in range(0,4):
	if i < 3:
		plt.subplot(2,2,i+1)
		plt.hist(betas_1[10000:,i], bins = 20, density = True)
		plt.xlabel(r'$\beta_{}$'.format(i))
		plt.title(r'$\beta_{} = {:.2f} \pm {:.2f}$'.format(i,np.mean(betas_1[10000:,i]),
			np.std(betas_1[10000:,i])))

	if i == 3:
		plt.subplot(2,2,i+1)
		plt.errorbar(x,y,yerr = sigma_y,label='Exp.',fmt = 'o')
		plt.plot(np.linspace(min(x),max(x),1000),model_a(np.linspace(min(x),max(x),1000),best_1),label='Fit')
		plt.title('BIC = {:.2f}'.format(BIC(best_1,3,x,y,sigma_y,1)))
		plt.xlabel('x')
		plt.ylabel('y')
		plt.legend()


plt.subplots_adjust(hspace=0.6,wspace=0.3)
plt.savefig('model_a.png')

plt.figure(figsize = (10,8))
for i in range(0,4):
	if i < 3:
		plt.subplot(2,2,i+1)
		plt.hist(betas_2[10000:,i], bins = 20, density = True)
		plt.xlabel(r'$\beta_{}$'.format(i))
		plt.title(r'$\beta_{} = {:.2f} \pm {:.2f}$'.format(i,np.mean(betas_2[10000:,i]),
			np.std(betas_2[10000:,i])))

	if i == 3:
		plt.subplot(2,2,i+1)
		plt.errorbar(x,y,yerr = sigma_y,label='Exp.',fmt = 'o')
		plt.plot(np.linspace(min(x),max(x),1000),model_b(np.linspace(min(x),max(x),1000),best_2),label='Fit')
		plt.title('BIC = {:.2f}'.format(BIC(best_2,3,x,y,sigma_y,2)))
		plt.xlabel('x')
		plt.ylabel('y')
		plt.legend()


plt.subplots_adjust(hspace=0.6,wspace=0.3)
plt.savefig('model_b.png')

plt.figure(figsize = (10,8))
for i in range(0,6):
	if i < 5:
		plt.subplot(2,3,i+1)
		plt.hist(betas_3[10000:,i], bins = 20, density = True)
		plt.xlabel(r'$\beta_{}$'.format(i))
		plt.title(r'$\beta_{} = {:.2f} \pm {:.2f}$'.format(i,np.mean(betas_3[10000:,i]),
			np.std(betas_3[10000:,i])))

	if i == 5:
		plt.subplot(2,3,i+1)
		plt.errorbar(x,y,yerr = sigma_y,label='Exp.',fmt = 'o')
		plt.plot(np.linspace(min(x),max(x),1000),model_c(np.linspace(min(x),max(x),1000),best_3),label='Fit')
		plt.title('BIC = {:.2f}'.format(BIC(best_3,5,x,y,sigma_y,3)))
		plt.xlabel('x')
		plt.ylabel('y')
		plt.legend()

plt.subplots_adjust(hspace=0.6,wspace=0.3)
plt.savefig('model_c.png')