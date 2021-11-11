import json
import xml.dom.minidom


# 读取json文件内容,返回字典格式
# with open('./answer.json', 'r', encoding='utf8')as fp:
#     json_data = json.load(fp)
#     print('这是文件中的json数据：', json_data)
#     print('这是读取到文件数据的数据类型：', type(json_data))
#     for every_img_ans in json_data:
#         # print(every_img_ans)
#         detect_info = every_img_ans["detect_info"]
#         img_name = every_img_ans["img_name"]
#         num_detInfo = len(detect_info)
#         print(num_detInfo)
#         if num_detInfo > 1:
#             print("-------------------")
#             break
#         for i in range(num_detInfo):
#             label = detect_info[i]["label"]  # str
#             score = detect_info[i]["score"]  # float
#             boxes = detect_info[i]["boxes"]  # list


# 测试方法一：将测试结果可视化展示
def visResult(testTxt, testImgDir, testAnnoDir):
    """
    将测试结果可视化
    :param testTxt:测试图片名称txt地址
    :param testImgDir: 测试图片地址
    :param testAnnoDir: 测试图片Anno地址
    :return:
    """
    with open(testTxt, 'r') as testTxt_f:
        testTxtLines = testTxt_f.readlines()
        for testTxtLine in testTxtLines:
            print(testTxtLine)
            testImgPath = testImgDir + "/" + testTxtLine + ".jpg"
            testAnnoPath = testAnnoDir + "/" + testTxtLine + ".xml"
            # dom变量
            curDom = xml.dom.minidom.parse(testAnnoPath)
            # 得到文档元素对象
            root = curDom.documentElement
            # 根据元素名字进行查找 object
            objects = root.getElementsByTagName('object')
            for j in range(len(objects)):
                object = objects[j]
                name = object.getElementsByTagName('name')[0]
                cls = name.firstChild.data
                # 确定不同类别的颜色
                if cls == "car":
                    color_turple = (0, 0, 0)
                elif cls == "truck":
                    color_turple = (255, 255, 0)
                elif cls == "bus":
                    color_turple = (255, 255, 255)


    return 0


testTxt = '/media/zzc/Backup Plus/数据集/车辆目标检测/临港相关数据集/LG/ImageSets/Main/test.txt'
testImgDir = '/media/zzc/Backup Plus/数据集/车辆目标检测/临港相关数据集/LG/JPEGImages'
testAnnoDir = '/media/zzc/Backup Plus/数据集/车辆目标检测/临港相关数据集/LG/Annotations'
visResult(testTxt, testImgDir, testAnnoDir)
