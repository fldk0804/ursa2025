import torch
import torch.nn as nn
import torch.optim as optim
import os

def supervised_train(args, model, loss_func, train_loader, val_loader):
    """
    Runs supervised training for the given model.
    
    Args:
        args: Parsed arguments containing hyperparameters and paths.
        model: The backbone model (CNN, ViT, etc.).
        loss_func: The loss function (e.g., CrossEntropyLoss).
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    num_epochs = 10  # Modify as needed

    log_path = os.path.join(args.exp_dir, "train_loss.log")
    with open(log_path, "w") as log_file:
        log_file.write("Epoch,Training Loss,Validation Loss\n")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = loss_func(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        with open(log_path, "a") as log_file:
            log_file.write(f"{epoch + 1},{avg_train_loss:.4f},{avg_val_loss:.4f}\n")

    print("Training Complete!")
