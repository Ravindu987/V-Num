model.fit(x=[training_img, train_padded_txt, train_input_length, train_label_length],
          y=np.zeros(len(training_img)),
          batch_size=batch_size, epochs=epochs,
          validation_data=([valid_img, valid_padded_txt, valid_input_length, valid_label_length],
          [np.zeros(len(valid_img))]), verbose=1, callbacks=callbacks_list)
