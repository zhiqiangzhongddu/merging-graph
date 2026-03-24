import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
from ..model.wrapper import get_model
from sklearn.linear_model import Ridge, LogisticRegression



def get_emb_y(loader, encoder, device, dtype='numpy'):
    x, y = encoder.get_emb(loader, device)
    if dtype == 'numpy':
        return x, y
    elif dtype == 'torch':
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    else:
        raise NotImplementedError


class EmbeddingEvaluation():
    def __init__(self, base_classifier, evaluator, task_type, num_tasks, device, params_dict=None,
                 param_search=True, metric='acc'):
        self.base_classifier = base_classifier
        self.evaluator = evaluator
        self.eval_metric = metric
        self.task_type = task_type
        self.num_tasks = num_tasks
        self.device = device
        self.param_search = param_search
        self.params_dict = params_dict
        if self.eval_metric == 'rmse':
            self.gscv_scoring_name = 'neg_root_mean_squared_error'
        elif self.eval_metric == 'mae':
            self.gscv_scoring_name = 'neg_mean_absolute_error'
        elif self.eval_metric == 'rocauc':
            self.gscv_scoring_name = 'roc_auc'
        elif self.eval_metric == 'accuracy':
            self.gscv_scoring_name = 'accuracy'
        else:
            raise ValueError('Undefined grid search scoring for metric %s ' % self.eval_metric)

        self.classifier = None

    def scorer(self, y_true, y_raw):
        input_dict = {"y_true": y_true, "y_pred": y_raw}
        score = self.evaluator.eval(input_dict)[self.eval_metric]
        return score

    def ee_binary_classification(self, train_emb, train_y, test_emb):
        if self.param_search:
            params_dict = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            self.classifier = make_pipeline(StandardScaler(),
                                            GridSearchCV(self.base_classifier, params_dict, cv=5,
                                                         scoring=self.gscv_scoring_name, n_jobs=16, verbose=0)
                                            )
        else:
            self.classifier = make_pipeline(StandardScaler(), self.base_classifier)

        self.classifier.fit(train_emb, np.squeeze(train_y))

        if self.eval_metric == 'accuracy':
            test_raw = self.classifier.predict(test_emb)
        else:
            test_raw = self.classifier.predict_proba(test_emb)[:, 1]

        return np.expand_dims(test_raw, axis=1)

    def ee_svc(self, train_emb, train_y, test_emb):
        if self.param_search:
            params_dict = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            self.classifier = make_pipeline(StandardScaler(),
                                            GridSearchCV(self.base_classifier, params_dict, cv=5,
                                                         scoring=self.gscv_scoring_name, n_jobs=16, verbose=0))
        else:
            self.classifier = make_pipeline(StandardScaler(), self.base_classifier)
        self.classifier.fit(train_emb, np.squeeze(train_y))
        test_raw = self.classifier.predict(test_emb)
        return np.expand_dims(test_raw, axis=1)

    def ee_multioutput_binary_classification(self, train_emb, train_y, test_emb):

        params_dict = {
            'multioutputclassifier__estimator__C': [1e-1, 1e0, 1e1, 1e2]}
        self.classifier = make_pipeline(StandardScaler(), MultiOutputClassifier(
            self.base_classifier, n_jobs=-1))

        if np.isnan(train_y).any():
            print("Has NaNs ... ignoring them")
            train_y = np.nan_to_num(train_y)
        self.classifier.fit(train_emb, train_y)

        test_raw = np.transpose([y_pred[:, 1] for y_pred in self.classifier.predict_proba(test_emb)])

        return test_raw

    def ee_regression(self, train_emb, train_y, val_emb, val_y, test_emb, test_y):
        if self.param_search:
            params_dict = {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]}
            # 			params_dict = {'alpha': [500, 50, 5, 0.5, 0.05, 0.005, 0.0005]}
            self.classifier = GridSearchCV(self.base_classifier, params_dict, cv=5,
                                           scoring=self.gscv_scoring_name, n_jobs=16, verbose=0)
        else:
            self.classifier = self.base_classifier

        self.classifier.fit(train_emb, np.squeeze(train_y))

        train_raw = self.classifier.predict(train_emb)
        val_raw = self.classifier.predict(val_emb)
        test_raw = self.classifier.predict(test_emb)

        return np.expand_dims(train_raw, axis=1), np.expand_dims(val_raw, axis=1), np.expand_dims(test_raw, axis=1)

    def embedding_evaluation(self, encoder, train_loader, test_loader):
        encoder.eval()
        train_emb, train_y = get_emb_y(train_loader, encoder, self.device)
        test_emb, test_y = get_emb_y(test_loader, encoder, self.device)
        if self.eval_metric == 'rocauc':
            if self.num_tasks == 1:
                test_raw = self.ee_binary_classification(train_emb, train_y, test_emb)
                test_score = self.scorer(test_y, test_raw)
        elif self.eval_metric == 'acc':
            test_raw = self.ee_svc(train_emb, train_y, test_emb)
            test_score = accuracy_score(test_y, test_raw)
        else:
            raise NotImplementedError
        return test_score

    def kf_embedding_evaluation(self, encoder, dataset, folds=10, batch_size=128):
        kf_train = []
        kf_val = []
        kf_test = []

        kf = KFold(n_splits=folds, shuffle=True, random_state=None)
        for k_id, (train_val_index, test_index) in enumerate(kf.split(dataset)):
            test_dataset = [dataset[int(i)] for i in list(test_index)]
            train_index, val_index = train_test_split(train_val_index, test_size=0.2, random_state=None)

            train_dataset = [dataset[int(i)] for i in list(train_index)]
            val_dataset = [dataset[int(i)] for i in list(val_index)]

            train_loader = DataLoader(train_dataset, batch_size=batch_size)
            valid_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            train_score, val_score, test_score = self.embedding_evaluation(encoder, train_loader, valid_loader,
                                                                           test_loader)

            kf_train.append(train_score)
            kf_val.append(val_score)
            kf_test.append(test_score)

        return np.array(kf_train).mean(), np.array(kf_val).mean(), np.array(kf_test).mean()

    # def linear_evaluation(self, args, model, dl_train, dl_test):


def run(args, device, model_name, init_model, dl_train, dl_test, evaluator=None):
    # model
    model = get_model(model_name, args, args.nclass).to(device)
    if hasattr(init_model, "project"):
        del init_model.project
    model.load_state_dict(init_model.state_dict(), strict=False)
    model.project = nn.Identity()
    score_list = []
    for i in range(1):
        if args.dataset in ['ogbg-molhiv', 'ogbg-molbbbp', 'ogbg-molbace', 'ogbg-molpcba']:
            ee = EmbeddingEvaluation(LogisticRegression(dual=False, fit_intercept=True, max_iter=5000),
                                     evaluator, args.task_type, args.num_tasks, device, params_dict=None,
                                     param_search=True, metric=args.metric)
            score = ee.embedding_evaluation(model, dl_train, dl_test)
            score_list.append(score)
        elif args.metric == 'rocauc':
            # Generic binary rocauc fallback without OGB evaluator
            model.eval()
            with torch.no_grad():
                X_train, Y_train = [], []
                for data in dl_train:
                    data = data.to(device)
                    x, edge_index, batch = data.x, data.edge_index, data.batch
                    emb = model(edge_index, x, batch)
                    X_train.append(emb.cpu())
                    Y_train.append(data.y.cpu())
                X_train = torch.cat(X_train, dim=0).numpy()
                Y_train = torch.cat(Y_train, dim=0).numpy()

                X_test, Y_test = [], []
                for data in dl_test:
                    data = data.to(device)
                    x, edge_index, batch = data.x, data.edge_index, data.batch
                    emb = model(edge_index, x, batch)
                    X_test.append(emb.cpu())
                    Y_test.append(data.y.cpu())
                X_test = torch.cat(X_test, dim=0).numpy()
                Y_test = torch.cat(Y_test, dim=0).numpy()
            if Y_train.ndim == 1 and X_train.shape[0] != Y_train.shape[0]:
                if Y_train.shape[0] % X_train.shape[0] != 0:
                    raise ValueError(
                        f"rocauc eval shape mismatch: X_train={X_train.shape} Y_train={Y_train.shape}"
                    )
                Y_train = Y_train.reshape(X_train.shape[0], -1)
            if Y_test.ndim == 1 and X_test.shape[0] != Y_test.shape[0]:
                if Y_test.shape[0] % X_test.shape[0] != 0:
                    raise ValueError(
                        f"rocauc eval shape mismatch: X_test={X_test.shape} Y_test={Y_test.shape}"
                    )
                Y_test = Y_test.reshape(X_test.shape[0], -1)
            if Y_train.ndim == 1 or (Y_train.ndim == 2 and Y_train.shape[1] == 1):
                Y_train = Y_train.reshape(-1)
                Y_test = Y_test.reshape(-1)
                clf = make_pipeline(StandardScaler(),
                                    LogisticRegression(dual=False, fit_intercept=True, max_iter=5000))
                clf.fit(X_train, Y_train)
                prob = clf.predict_proba(X_test)[:, 1]
                score = roc_auc_score(Y_test, prob)
                score_list.append(score)
            else:
                scores = []
                for i in range(Y_train.shape[1]):
                    y_tr = Y_train[:, i]
                    y_te = Y_test[:, i]
                    mask_tr = ~np.isnan(y_tr)
                    mask_te = ~np.isnan(y_te)
                    if not np.any(mask_tr) or not np.any(mask_te):
                        continue
                    y_tr = y_tr[mask_tr]
                    y_te = y_te[mask_te]
                    if set(np.unique(y_tr)) == {-1, 1} or set(np.unique(y_te)) == {-1, 1}:
                        y_tr = (y_tr > 0).astype(int)
                        y_te = (y_te > 0).astype(int)
                    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                        continue
                    clf = make_pipeline(StandardScaler(),
                                        LogisticRegression(dual=False, fit_intercept=True, max_iter=5000))
                    clf.fit(X_train[mask_tr], y_tr)
                    prob = clf.predict_proba(X_test[mask_te])[:, 1]
                    scores.append(roc_auc_score(y_te, prob))
                if not scores:
                    raise ValueError("rocauc evaluation failed: no valid tasks with both classes.")
                score_list.append(float(np.mean(scores)))
        elif args.metric in {'rmse', 'mae'}:
            model.eval()
            with torch.no_grad():
                X_train, Y_train = [], []
                for data in dl_train:
                    data = data.to(device)
                    x, edge_index, batch = data.x, data.edge_index, data.batch
                    emb = model(edge_index, x, batch)
                    X_train.append(emb.cpu())
                    Y_train.append(data.y.cpu())
                X_train = torch.cat(X_train, dim=0).numpy()
                Y_train = torch.cat(Y_train, dim=0).numpy()

                X_test, Y_test = [], []
                for data in dl_test:
                    data = data.to(device)
                    x, edge_index, batch = data.x, data.edge_index, data.batch
                    emb = model(edge_index, x, batch)
                    X_test.append(emb.cpu())
                    Y_test.append(data.y.cpu())
                X_test = torch.cat(X_test, dim=0).numpy()
                Y_test = torch.cat(Y_test, dim=0).numpy()

            if Y_train.ndim == 1:
                Y_train = Y_train.reshape(-1, 1)
            if Y_test.ndim == 1:
                Y_test = Y_test.reshape(-1, 1)

            reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            reg.fit(X_train, Y_train)
            pred = reg.predict(X_test)
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)

            if args.metric == 'rmse':
                scores = [
                    float(np.sqrt(mean_squared_error(Y_test[:, i], pred[:, i])))
                    for i in range(Y_test.shape[1])
                ]
            else:
                scores = [
                    mean_absolute_error(Y_test[:, i], pred[:, i])
                    for i in range(Y_test.shape[1])
                ]
            score_list.append(float(np.mean(scores)))

            # model.eval()
            # with torch.no_grad():
            #     # tr feature
            #     X_train, Y_train = [], []
            #     for data in dl_train:
            #         data = data.to(device)
            #         x, edge_index, = data.x, data.edge_index
            #         batch = data.batch
            #         emb = model(edge_index, x, batch)
            #         X_train.append(emb)
            #         Y_train.append(data.y)
            #     X_train, Y_train = torch.cat(X_train, dim=0), torch.cat(Y_train, dim=0)
            #     num_features = X_train.shape[-1]
            #     loader_emb_train = DataLoader(TensorDataset(X_train, Y_train), batch_size=args.test_bs,
            #                                   shuffle=True)
            #
            #     # te feature
            #     X_test, Y_test = [], []
            #     for data in dl_test:
            #         data = data.to(args.device)
            #         x, edge_index, = data.x, data.edge_index
            #         batch = data.batch
            #         emb = model(edge_index, x, batch)
            #         X_test.append(emb)
            #         Y_test.append(data.y)
            #     X_test, Y_test = torch.cat(X_test, dim=0), torch.cat(Y_test, dim=0)
            #     loader_emb_test = DataLoader(TensorDataset(X_test, Y_test), batch_size=args.test_bs,
            #                                  shuffle=False)
            #
            #     pred_head = nn.Sequential(nn.Linear(num_features, 1)).to(device)
            #     opt = torch.optim.Adam(pred_head.parameters(), lr=args.test_lr, weight_decay=args.test_wd)
            #     cls_criterion = torch.nn.BCEWithLogitsLoss()
            #
            #     pred_head.train()
            #     loss_all = 0
            #     for _ in range(args.test_epoch):
            #         for x, y in loader_emb_train:
            #             y = y.view(-1, 1).float()
            #             opt.zero_grad()
            #             output = pred_head(x)
            #             loss = cls_criterion(output, y)
            #             loss.requires_grad = True
            #             loss.backward()
            #             opt.step()
            #             loss_all += loss.item() * y.size(0)
            #     loss = loss_all / len(loader_emb_train)
            #     print(f'Evaluation Stage - loss: {loss:.4f}')
            #
            #     pred_head.eval()
            #     pred, y_true = [], []
            #     with torch.no_grad():
            #         for x, y in loader_emb_test:
            #             output = pred_head(x)
            #             pred.append(output)
            #             y_true.append(y.view(-1, 1))
            #     score_test = evaluator.eval({'y_pred': torch.cat(pred),
            #                                  'y_true': torch.cat(y_true)})['rocauc']
            #     score_list.append(score_test)
            # return np.mean(score_list), np.std(score_list)

        elif str(args.dataset).lower() in {'proteins', 'nci1', 'dd', 'nci109'}:
            model.eval()
            with torch.no_grad():
                # tr feature
                X_val, Y_val = [], []
                for data in dl_train:
                    data = data.to(args.device)
                    x, edge_index, = data.x, data.edge_index
                    batch = data.batch
                    emb = model(edge_index, x, batch)
                    X_val.append(emb)
                    Y_val.append(data.y)
                X_val, Y_val = torch.cat(X_val, dim=0), torch.cat(Y_val, dim=0)
                num_features = X_val.shape[-1]
                loader_emb_val = DataLoader(TensorDataset(X_val, Y_val), batch_size=args.test_bs,
                                            shuffle=True, num_workers=0, )

                # te feature
                X_test, Y_test = [], []
                for data in dl_test:
                    data = data.to(args.device)
                    x, edge_index, = data.x, data.edge_index
                    batch = data.batch
                    emb = model(edge_index, x, batch)
                    X_test.append(emb)
                    Y_test.append(data.y)
                X_test, Y_test = torch.cat(X_test, dim=0), torch.cat(Y_test, dim=0)
                loader_emb_test = DataLoader(TensorDataset(X_test, Y_test), batch_size=args.test_bs,
                                             shuffle=False, num_workers=0, )

            # -------------------------------------------------------------------------------------------------------------------------------------------------------#

            """LINEAR EVALUATION"""

            pred_head = nn.Linear(num_features, args.nclass).to(args.device)
            opt = torch.optim.Adam(pred_head.parameters(), lr=args.test_lr, weight_decay=args.test_wd)

            # print("Linear Evaluation")
            pred_head.train()
            for _ in range(args.test_epoch):
                for x, y in loader_emb_val:
                    loss = F.nll_loss(torch.log_softmax(pred_head(x), dim=1), y.view(-1))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            pred_head.eval()
            with torch.no_grad():
                for _ in range(5):
                    meta_loss, meta_acc, denominator = 0., 0., 0.
                    true_list = []
                    pred_list = []
                    for x, y in loader_emb_test:
                        l = pred_head(x)
                        meta_loss += F.nll_loss(torch.log_softmax(l, dim=1), y.view(-1), reduction="sum")
                        pred = l.argmax(dim=-1)
                        pred_list.append(pred.cpu().numpy())
                        true_list.append(y.view(-1).cpu().numpy())
                    #     meta_acc += torch.eq(pred, y.view(-1)).float().sum()
                    #     denominator += x.shape[0]
                    # meta_loss /= denominator
                    # meta_acc /= denominator
                    true_list = np.concatenate(true_list, 0)
                    pred_list = np.concatenate(pred_list, 0)
                    score = accuracy_score(true_list, pred_list)
                    score_list.append(score)
        else:
            raise NotImplementedError
    return np.mean(score_list), np.std(score_list)
