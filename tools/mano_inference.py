import tyro

from wilor_nano.api.mano_inference import ManoConfig, main

if __name__ == "__main__":
    main(tyro.cli(ManoConfig))
