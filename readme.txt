環境需求如 requirements.txt，可用 pip install -r requirements.txt 一鍵安裝
執行程式前先開啟環境 source 路徑/requirements.txt

執行每個程式 python3 路徑/main.py 參數1 參數2 參數3
參數數量不一定，請看程式碼本身，資料擴增方法的參數為固定
    method = int(sys.argv[1])
    0:original, 1:smote, 2:adasyn
    ex: python3 ./main.py 2    代表使用adasyn進行資料擴增

NSLKDD資料夾包含原始資料與one-hot編碼完的資料
get_data.py 負責把編碼完的資料導入
ave.py 計算平均結果 ex: ave.py 路徑
wilcoxon.py 計算魏克生顯著差異
ablation 資料夾為消融實驗
LU_tensor 資料夾為季澤學長的方法(ASEA)，但不包含自適應調整超參數的部分，主要可看當中的augmentation資料夾

multi_layer 資料夾 (proposed method)
    deal_label.py 算是編碼過程
    許多機器學習命名的資料夾為實驗第二部分不同方法的結合 (論文中的機器學習實驗)，RF的實驗結果則放在result資料夾，svm跑太久沒跑了

/home/wang/Desktop/thesis/SVM/main.py 當中程式碼註解的部分為計算AUC/ROC，論文後續沒採用這兩項指標，往後需要可使用
