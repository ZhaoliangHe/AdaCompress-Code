import urllib
import json
import base64
from io import BytesIO
from ImageCompressionEnvironment import EnvironmentAPI


class Baidu(object):
    def __init__(self,
                 AK='gVdv8eOhIkPKSnQVgRscENWA',
                 SK='OYaTkZxjLFAEdTdBewCxuKPF8jX8HKc8'):
        self.api_name = "baidu"
        self.ak = AK
        self.sk = SK

        self.token = self._get_access_token()

    def _get_access_token(self):
        host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=%s&client_secret=%s' % \
               (self.ak, self.sk)
        request = urllib.request.Request(host)
        request.add_header('Content-Type', 'application/json; charset=UTF-8')
        response = urllib.request.urlopen(request)
        content = response.read().decode()
        return json.loads(content)['access_token']

    def _base64_encode(self, image, quality):
        f = BytesIO()
        image.save(f, format='jpeg', quality=quality)
        binary_data = f.getvalue()
        f.seek(0)
        size = len(f.getvalue())
        return base64.b64encode(binary_data), size

    def recognize(self, image, quality):
        img_b64, size = self._base64_encode(image, quality)
        request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v2/advanced_general"
        params = {"image": img_b64}
        params = urllib.parse.urlencode(params).encode(encoding='UTF8')

        access_token = self.token
        request_url = request_url + "?access_token=" + access_token
        request = urllib.request.Request(url=request_url, data=params)
        request.add_header('Content-Type', 'application/x-www-form-urlencoded')
        response = urllib.request.urlopen(request)
        content = response.read().decode()
        response_dict = json.loads(content)
        if not 'error_code' in response_dict.keys():
            return 0, response_dict['result'], size
        else:
            print(response_dict['error_msg'])
            return 1, [response_dict['error_msg']], size


class FacePP(Baidu):
    def __init__(self, AK='kfUXcnJUAEWr2kv3CGYWV7ND56VO6Mr3', SK='ta0I1ms02Txt70wulHWTALu7anDHoDLP'):
        self.api_name = "face_plusplus"
        self.ak = AK
        self.sk = SK

    def recognize(self, image, quality):
        img_b64, size = self._base64_encode(image, quality)
        request_url = "https://api-cn.faceplusplus.com/imagepp/beta/detectsceneandobject"
        params = {"api_key": self.ak,
                  "api_secret": self.sk,
                  "image_base64": img_b64}
        params = urllib.parse.urlencode(params).encode(encoding="UTF-8")
        request = urllib.request.Request(url=request_url, data=params)
        response = urllib.request.urlopen(request)
        content = response.read().decode()
        response_dict = json.loads(content)

        if not "error_message" in response_dict.keys():
            if len(response_dict['objects']) == 0:
                return 2, [{"keyword": "", "score": 1e-6}], size
            result_dicts = [{"keyword": line_dict['value'], "score": line_dict['confidence']} for line_dict in response_dict['objects']]
            return 0, result_dicts, size
        else:
            print(response_dict['error_message'])
            return 1, [response_dict['error_message']], size


# class Algorithmia(object):



if __name__ == '__main__':
    face_pp = FacePP()
    baidu = Baidu()
    env = EnvironmentAPI(imagenet_train_path='/home/hsli/imagenet-data/train/',
                         samples_per_class=13,
                         cloud_agent=face_pp)

    face_pp.recognize(env.image_datalist[2], 75)