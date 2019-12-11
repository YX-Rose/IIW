def select_model(rec_model=''):
    if rec_model == 'VGGFace':
        from models.recognition.VGGFace import VGGFace, extract_feature
        feature_extractor = VGGFace(model_path='../../pretrained/VGGFace/model.pth.tar')
        extract_feature_import = extract_feature
    elif rec_model == 'ArcFace':
        from models.recognition.ArcFace import ArcFace, extract_feature
        feature_extractor = ArcFace(model_path='../../pretrained/ArcFace/model.pth.tar')
        extract_feature_import = extract_feature
    elif rec_model == 'SphereFace':
        from models.recognition.SphereFace import SphereFace, extract_feature
        feature_extractor = SphereFace(model_path='../../pretrained/SphereFace/sphere20a_20171020.pth')
        extract_feature_import = extract_feature
    elif rec_model == 'MobileFace':
        from models.recognition.MobileFace import MobileFace, extract_feature
        feature_extractor = MobileFace(model_path='../../pretrained/MobileFace/model.pth.tar')
        extract_feature_import = extract_feature
    elif rec_model == 'LightCNN_9':
        from models.recognition.LightCNN_9 import LightCNN_9, extract_feature
        feature_extractor = LightCNN_9(model_path='../../pretrained/LightCNN_9/model.pth.tar')
        extract_feature_import = extract_feature
    elif rec_model == 'LightCNN_29v2':
        from models.recognition.LightCNN_29v2 import LightCNN_29v2, extract_feature
        feature_extractor = LightCNN_29v2(model_path='../../pretrained/LightCNN_29v2/model.pth.tar')
        extract_feature_import = extract_feature

    return feature_extractor, extract_feature_import