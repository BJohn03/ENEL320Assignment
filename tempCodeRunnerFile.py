result[image.shape[0] // 2 + 1][image.shape[1] // 2 - 1] = 1
    result[image.shape[0] // 2 + 2][image.shape[1] // 2 - 2] = 1
    result[image.shape[0] // 2 + 3][image.shape[1] // 2 - 3] = 1
    result[image.shape[0] // 2 + 4][image.shape[1] // 2 - 4] = 1
    result[image.shape[0] // 2 + 5][image.shape[1] // 2 - 5] = 1
    result[image.shape[0] // 2 + 6][image.shape[1] // 2 - 6] = 1
    result[image.shape[0] // 2 + 7][image.shape[1] // 2 - 7] = 1
    result[image.shape[0] // 2 + 8][image.shape[1] // 2 - 8] = 1
    result[image.shape[0] // 2 + 9][image.shape[1] // 2 - 9] = 1
    result[image.shape[0] // 2 + 1 ][image.shape[1] // 2 - 10 : image.shape[1] // 2 + 10] = 0.1
    result[image.shape[0] // 2 - 1 ][image.shape[1] // 2 - 10 : image.shape[1] // 2 + 10] = 0.1
