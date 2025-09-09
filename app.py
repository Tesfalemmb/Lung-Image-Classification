if uploaded_file is not None:
    img = Image.open(uploaded_file)
    if model is None:
        st.error("Model failed to load. Please check the model file.")
    else:
        img_array = preprocess_image(img)
        if img_array is None:
            st.error("Failed to process image")
            return
        
        preds = model.predict(img_array, verbose=0)[0]
        pred_class_index = np.argmax(preds)
        pred_class = class_names[pred_class_index]
        confidence = np.max(preds) * 100
        
        # Grad-CAM overlay
        heatmap = get_gradcam(img_array, model, pred_class_index)
        if heatmap is not None:
            # Resize heatmap to match original image
            heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            img_np = np.array(img.convert('RGB'))
            superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
        else:
            superimposed_img = np.array(img.convert('RGB'))
        
        # Display original and overlay side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Image", use_column_width=True)
        with col2:
            st.image(superimposed_img, caption=f"Grad-CAM Overlay for {pred_class}", use_column_width=True)
        
        # Prediction bars below
        st.subheader("📊 Prediction Results")
        for i, (class_name, prob) in enumerate(zip(class_names, preds)):
            color = (
                "green" if class_name == "Healthy" else
                "red" if class_name == "Inflammation" else
                "blue" if class_name == "Neoplastic" else
                "orange"
            )
            st.markdown(f"**<span style='color: {color}'>{class_name}:</span> {prob*100:.2f}%**", unsafe_allow_html=True)
            st.progress(float(prob))
        
        prediction_color = (
            "green" if pred_class == "Healthy" else
            "red" if pred_class == "Inflammation" else
            "blue" if pred_class == "Neoplastic" else
            "orange"
        )
        st.markdown(f"<h3 style='color: {prediction_color}'>✅ Final Prediction: {pred_class}</h3>", unsafe_allow_html=True)
        st.info(f"**Confidence: {confidence:.2f}%**")
