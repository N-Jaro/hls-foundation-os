from mmseg.datasets.pipelines import LoadImageFromFile
import json

class LoadImageWithMetadata(LoadImageFromFile):
    def __call__(self, results):
        filename = results['filename']
        print(filename)
        # img = super().__call__(results)  # Use default image loading

        # # Example: Load metadata from a JSON file associated with the filename
        # meta_filename = filename.replace('.jpg', '.json') 
        # with open(meta_filename) as f:
        #     metadata = json.load(f)

        # results['img'] = img
        # results['metadata'] = metadata         
        return results