import os

URL = 'http://thecellmap.org/costanzo2016/data_files/Raw%20genetic%20interaction%20datasets:%20Matrix%20format.zip'
output_dir = 'data'
raw_zip = os.path.join(output_dir, 'raw/costanzo_supplement.zip')

rule download:
    params:
        URL
    output:
        os.path.join(output_dir, 'raw/costanzo_supplement.zip')
    shell:
        '''
        wget {params} -O {output}
        '''

rule unzip:
    input:
        raw_zip
    output:
        directory('data/output')
    shell:
        '''
        unzip {input} -d {output}
        '''

rule process_gis:
    