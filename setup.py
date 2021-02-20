from setuptools import setup

setup(name='experiment_runner',
      version='0.1',
      description='Simple wrapper to run multiple experiments via Ray / Multiprocessing on different machines',
      url='https://github.com/sbuschjaeger/experiment_runner/',
      author=u'Sebastian Buschjäger, Lukas Pfahler',
      author_email='{sebastian.buschjaeger, lukas.pfahler}@tu-dortmund.de',
      license='MIT',
      packages=['experiment_runner'],
      zip_safe=False,
      setup_requires=[
            "numpy",
            "tqdm"
      ],
      extras_require={
            "ray": ["ray"],
            "malocher": ["malocher @ git+https://github.com/Whadup/malocher@main#egg=malocher"]
      }
)
